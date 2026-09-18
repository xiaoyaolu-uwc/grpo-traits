import json
import argparse
import time
from datetime import datetime
import statistics as stats
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_traits import core, rollouts, reward, metrics, prompts
from grpo_traits.evaluate import eval_model

# ---------- default parameter constants ----------
MODEL_ID = "Qwen/Qwen3-0.6B"
B = 1                      # prompts per step
G = 8                      # rollouts per prompt
MAX_NEW = 256
LR = 1e-5
STEPS = 20
AGGREGATION = "max_length"
TEMP = 0.7

IS_STRICT = False
GRAD_CLIP = 1.0
SEED = 0
STD_CORRECT = True         # Original GRPO vs. DR. GRPO

DATA_DIR = Path(__file__).parent / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
EVAL_PATH = DATA_DIR / "test.jsonl"
LOG_DIR = Path(__file__).parent.parent.parent / "logs"

# ---------- Data loading functions ----------
def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def next_batch(step, answerable_rows, unanswerable_rows, size=B):
    """Returns a list of training rows, alternating between answerable and unanswerable rows."""
    pool = answerable_rows if step % 2 == 0 else unanswerable_rows
    start = (step // 2) * size
    picked = [pool[(start + j) % len(pool)] for j in range(size)]
    return picked

def main(run_name, steps, group_size, max_new, temp, is_strict, time_now):
    # ---------- Training setup ----------
    # Seeding
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    model.to(device)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR)

    # ---------- Data ----------
    train_rows = load_rows(TRAIN_PATH)
    eval_rows = load_rows(EVAL_PATH)
    answerable_rows = [r for r in train_rows if r["answerable"]]
    unanswerable_rows = [r for r in train_rows if not r["answerable"]]
    random.shuffle(answerable_rows)
    random.shuffle(unanswerable_rows)

    # ---------- Logging ----------
    train_logger = metrics.MetricsLogger(run_name, time_now, metrics.TRAIN_FIELDS, suffix="train")
    eval_logger  = metrics.MetricsLogger(run_name, time_now, metrics.EVAL_FIELDS, suffix="eval")
    sample_logger = metrics.SampleLogger(run_name, time_now)

    # Log configuration of run
    sample_logger.log_config(
        run_name=run_name,
        model_id=MODEL_ID,
        batch_size=B,
        group_size=group_size,
        steps=steps,
        max_new=max_new,
        temp=temp,
        lr=LR,
        aggregation=AGGREGATION,
        is_strict=is_strict,
        std_correct=STD_CORRECT,
        grad_clip=GRAD_CLIP,
        seed=SEED,
        system_prompt=prompts.SYSTEM_PROMPT,
    )

    # Baseline eval
    baseline_stats = eval_model(model, tokenizer, rows=eval_rows, max_new=max_new, temp=temp)
    eval_logger.log_metrics(
        step=0,
        **baseline_stats,
    )

    # ---------- training loop ----------
    for step in range(steps):
        optimizer.zero_grad()
        # Get rows and build prompts
        rows = next_batch(step, answerable_rows, unanswerable_rows)
        questions = [row["question"] for row in rows]
        inputs = [prompts.build_prompt(question, tokenizer) for question in questions]
        print(questions)

        # Run rollouts
        generation_start = time.perf_counter()
        tokenized_prompts = rollouts.tokenize_prompts(inputs, tokenizer).to(device)
        completion_ids = rollouts.generate_rollouts(model=model, tokenized_prompts=tokenized_prompts, 
                                                    max_new=max_new, group_size=group_size, temp=temp)
        decoded_completions = tokenizer.batch_decode(
            completion_ids[:, tokenized_prompts.shape[-1]:],
            skip_special_tokens=True
        )
        print("completions: ", decoded_completions)

        # Compute reward and answer stats
        answers = reward.extract_answers(decoded_completions)
        rewards, answer_stats = reward.compute_reward(answers=answers, group_size=group_size, 
                                                      rows=rows, is_strict=is_strict)
        reward_std = torch.tensor(rewards, dtype=torch.float32).view(B, group_size).std(dim=-1).mean().item()
        print("rewards: ", rewards)

        # Advantage and advantage statistics
        completion_mask = core.completion_mask(completions=completion_ids, max_prompt_length=tokenized_prompts.shape[-1], pad_id=tokenizer.pad_token_id)
        advantages = core.compute_advantage(rewards=rewards, group_size=group_size,
                                            std_correct=STD_CORRECT).to(device)
        print("advantages: ", advantages)
        adv_t = advantages.view(B, group_size)
        adv_avg = adv_t.mean().item()
        adv_std = adv_t.std(dim=-1).mean().item()

        # Logprobs, loss, and gradient. Skip if step is degenerate
        if adv_std > 0:
            log_probs = rollouts.rollout_logprobs(model=model, completions=completion_ids)
            loss = core.compute_loss(advantages=advantages, log_probs=log_probs, 
                                     completion_mask=completion_mask, max_new=max_new, aggregation=AGGREGATION)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_value = loss.item()
        else:
            loss_value = None
        print("loss: ", loss_value)
        print("adv_std: ", adv_std)

        # Compute avg response length
        resp_lengths = completion_mask.sum(dim=-1).float()     
        avg_response_lengths = resp_lengths.mean().item()

        # Compute tokens per second
        if device.type == "mps":
            torch.mps.synchronize()
        step_secs = time.perf_counter() - generation_start
        tokens_per_sec = resp_lengths.sum().item() / step_secs

        # Log metrics
        train_logger.log_metrics(
            step=step,
            total_steps=steps,
            loss=loss_value,
            reward_avg=stats.mean(rewards),
            reward_std=reward_std,
            tokens_per_sec=tokens_per_sec,
            avg_response_len=avg_response_lengths,
            adv_avg=adv_avg,
            adv_std=adv_std,
            **answer_stats,
            kl_loss=None,
            policy_ratio=None,
            entropy_avg=None,
        )

        # Log responses 
        sample_logger.log_samples(
            step=step, 
            rows=rows, 
            answers=answers, 
            rewards=rewards, 
            completions=decoded_completions, 
            group_size=group_size)


    # ---------- final eval ----------
    final_stats = eval_model(model, tokenizer, rows=eval_rows, max_new=max_new, temp=temp)
    eval_logger.log_metrics(
        step=steps,
        **final_stats,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="run")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--group_size", type=int, default=G)
    p.add_argument("--max_new", type=int, default=MAX_NEW)
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    main(
        run_name=args.run_name,
        steps=args.steps,
        group_size=args.group_size,
        max_new=args.max_new,
        temp=args.temp,
        is_strict=args.strict,
        time_now=now
    )