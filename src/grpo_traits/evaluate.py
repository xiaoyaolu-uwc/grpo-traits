from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo_traits import reward, rollouts
from grpo_traits.prompts import build_prompt

def eval_model(model, tokenizer, rows, max_new, temp, chunk_size=8, is_strict=False):
    """
    Compute statistics over 
    """
    was_training = model.training
    model.eval()
    totals = {}
    try:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start+chunk_size]

            # Build prompts for model
            prompts = [build_prompt(row["question"], tokenizer) for row in chunk]
            tokenized_prompts = rollouts.tokenize_prompts(prompts, tokenizer).to(model.device)
            
            # Run rollouts on prompts
            completion_ids = rollouts.generate_rollouts(model=model, tokenized_prompts=tokenized_prompts, 
                                                        max_new=max_new, group_size=1, do_sample=False, 
                                                        temp=temp)
            decoded_completions = tokenizer.batch_decode(
                        completion_ids[:,tokenized_prompts.shape[-1]:],
                        skip_special_tokens=True
                    )
            
            # Score rollouts and return
            answers = reward.extract_answers(decoded_completions)
            rewards, answer_stats = reward.compute_reward(answers=answers, group_size=1, rows=chunk, is_strict=is_strict)
            for k, v in answer_stats.items():
                totals[k] = totals.get(k, 0) + v

        totals["avg_acc"] = (
            totals["answered_of_answerable"] + totals["abstained_of_unanswerable"]
        ) / (totals["num_answerable"] + totals["num_unanswerable"])

    finally:
        model.train(was_training)

    return totals 
