from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo_traits import reward, rollouts
from grpo_traits.prompts import build_prompt
import statistics as stats


def eval_model(model, tokenizer, rows, max_new, chunk_size=8, is_strict=False):
    """
    Compute statistics over 
    """
    model.eval()
    avg_accuracy = 0
    totals = {}
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start+chunk_size]

        # Build prompts for model
        prompts = [build_prompt(row["question"], tokenizer) for row in chunk]
        tokenized_prompts = rollouts.tokenize_prompts(prompts, tokenizer)
        
        # Run rollouts on prompts
        completion_ids = rollouts.generate_rollouts(model=model, 
        tokenized_prompts=tokenized_prompts, max_new= max_new, group_size=1)
        decoded_completions = tokenizer.batch_decode(
                    completion_ids,
                    skip_special_tokens=True
                )
        
        # Score rollouts and return
        answers = reward.extract_answers(decoded_completions)
        rewards, answer_stats = reward.compute_reward(answers=answers, group_size=1, rows=rows, is_strict=is_strict)
        avg_accuracy += answers
        for k, v in answer_stats.items():
            totals[k] = totals.get(k, 0) + v
    avg_accuracy /= len(rows)
    return avg_accuracy, totals 
