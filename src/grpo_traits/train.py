import json
import time
import statistics as stats
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_traits import core, rollouts, reward, prompts, metrics

# ---------- constants ----------
MODEL_ID = "Qwen/Qwen3-0.6B"
B = 1                      # prompts per step
G = 4                      # rollouts per prompt
MAX_NEW = 64
LR = 1e-5
STEPS = 20
AGGREGATION = "max_length"
IS_STRICT = False
GRAD_CLIP = 1.0
SEED = 0
STD_CORRECT = True         # Original GRPO vs. DR. GRPO

DATA_DIR = Path(__file__).parent / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
LOG_DIR = Path(__file__).parent.parent.parent / "logs"

# ---------- data ----------
def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

rows = load_rows(TRAIN_PATH)
answerable_rows = [r for r in rows if r["answerable"]]
unanswerable_rows = [r for r in rows if not r["answerable"]]
random.shuffle(answerable_rows)
random.shuffle(unanswerable_rows)

def next_batch(step, size=B):
    """Returns a list of training rows, alternating between answerable and unanswerable rows."""
    pool = answerable_rows if step % 2 == 0 else unanswerable_rows
    start = (step // 2) * size
    picked = [pool[(start + j) % len(pool)] for j in range(size)]
    return picked



def main(run_name = ""):
    # ---------- setup ----------
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

    # Intitialize metrics logger
    metrics_logger = metrics.MetricsLogger(run_name)

    # ---------- training loop ----------
    for step in range(STEPS):
        optimizer.zero_grad()
        # Get rows and build prompts
        rows = next_batch(step)
        questions = [row["question"] for row in rows]
        prompts = [prompts.build_prompt(question) for question in questions]

        # Run rollouts
        generation_start = time.perf_counter()
        tokenized_prompts = rollouts.tokenize_prompts(prompts, tokenizer)
        completion_ids = rollouts.generate_rollouts(model=model, tokenized_prompts=tokenized_prompts, max_new=MAX_NEW, group_size=G)
        decoded_completions = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True
        )

        # Compute reward and answer stats
        answers = reward.extract_answers(decoded_completions)
        rewards, answer_stats = reward.compute_reward(answers=answers, group_size=G, rows=rows, is_strict=IS_STRICT)

        # Advantage and log_probs
        completion_mask = core.completion_mask(completions=completion_ids, max_prompt_length=MAX_NEW, pad_id=tokenizer.pad_token_id)
        advantages = core.compute_advantage(rewards=rewards, group_size=G, std_correct=STD_CORRECT)
        log_probs = rollouts.rollout_logprobs(model=model, completions=completion_ids)

        # Compute loss
        loss = core.compute_loss(advantages=advantages, log_probs=log_probs, completion_mask=completion_mask)

        # Backward 
        loss.backward()
        optimizer.step()

        # Compute metrics
        resp_lengths = completion_mask.sum(dim=-1).float()     
        avg_response_lengths = resp_lengths.mean().item()
        if device.type == "mps":
            torch.mps.synchronize()
        step_secs = time.perf_counter() - generation_start
        tokens_per_sec = generation_start.sum().item() / step_secs
        adv_t = advantages.view(B, G)
        adv_avg = adv_t.mean().item()
        adv_std = adv_t.std(dim=-1).mean().item()

        # Report metrics
        metrics_logger.log_metrics(
            step=step,
            total_steps=STEPS,
            loss=loss,
            reward_avg=stats.mean(rewards),
            tokens_per_sec=tokens_per_sec,
            avg_response_len=avg_response_lengths,
            adv_avg=adv_avg,
            adv_std=adv_std,
            eval_acc,
            **answer_stats,
            kl_loss=None,
            policy_ratio=None,
            entropy_avg=None,
        )
