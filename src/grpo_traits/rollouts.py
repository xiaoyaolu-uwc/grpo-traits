from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def tokenize_prompts(prompts: list, tokenizer: AutoTokenizer) -> torch.Tensor:
    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    max_token_length = max(len(prompt_tokens) for prompt_tokens in tokenized_prompts)
    pad_id = tokenizer.pad_token_id
    padded_prompts = []
    attention_mask = []
    for prompt in tokenized_prompts:
        # Build padded prompts
        pad_length = max_token_length - len(prompt)
        padding = [pad_id] * pad_length
        padded_prompt = padding + prompt
        padded_prompts.append(padded_prompt)
        # Build attention mask
        padding_indic = [0] * pad_length
        prompt_indic = [1] * len(prompt)
        mask = padding_indic + prompt_indic
        attention_mask.append(mask)
    return torch.tensor(padded_prompts), torch.tensor(attention_mask)

@torch.no_grad
def generate_rollouts(model: AutoModelForCausalLM, tokenized_prompts: torch.Tensor, 
                      max_new: int, group_size: int, temp, prompt_mask: torch.Tensor, 
                      do_sample: bool = True) -> torch.Tensor:
    """
    Produce completion tensor, shape (batch * group_size, max_completion_length)
    """
    completions = model.generate(tokenized_prompts, max_new_tokens=max_new, 
                                 num_return_sequences=group_size, do_sample=do_sample,
                                 temperature=temp, attention_mask=prompt_mask)
    return completions

def rollout_logprobs(model: AutoModelForCausalLM, completions: torch.Tensor, 
                     prompt_mask: torch.Tensor):
    """
    Produce tensor of logprobs for each completion, shape (batch * group_size, 
    max_completion_length - 1). There are no log probs for the first token.
    """
    # Build mask for left padding
    batch, prompt_len = prompt_mask.shape
    rows = completions.shape[0]
    group_size = rows // batch
    gen_len = completions.shape[-1] - prompt_len
    full_mask = torch.cat(
        [
            prompt_mask.repeat_interleave(group_size, dim=0),
            torch.ones(rows, gen_len, dtype=prompt_mask.dtype,
                        device=prompt_mask.device),
        ],
        dim=1,
    )

    # Get logits over completions
    logits = model(completions, attn_mask=full_mask).logits
    # Convert logits to log probaiblities
    log_probs_over_volcab = torch.log_softmax(logits, dim=-1)
    # Truncate probs and shift tokens for indexing probabilities
    prediction_log_probs = log_probs_over_volcab[:, :-1, :]
    target_tokens = completions[:, 1:]
    # Gather logprobs
    target_log_probs = prediction_log_probs.gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    return target_log_probs