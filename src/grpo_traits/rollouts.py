from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def tokenize_prompts(prompts: list, tokenizer: AutoTokenizer) -> torch.Tensor:
    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    max_token_length = max(len(prompt_tokens) for prompt_tokens in tokenized_prompts)
    pad_id = tokenizer.pad_token_id
    padded_prompts = []
    for prompt in tokenized_prompts:
        pad_length = max_token_length - len(prompt)
        padding = [pad_id] * pad_length
        padded_prompt = padding + prompt
        padded_prompts.append(padded_prompt)
    return torch.tensor(padded_prompts)

@torch.no_grad
def generate_rollouts(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list, 
                      max_new: int, num_rollouts: int) -> torch.Tensor:
    """
    Produce completion tensor, shape (batch, num_rollouts, max_completion_length)
    """
    tokenized_prompts = tokenize_prompts(prompts, tokenizer).to(model.device)
    completions = model.generate(tokenized_prompts, max_new_tokens=max_new, 
                                 num_return_sequences=num_rollouts)
    return completions

def rollout_logprobs(model: AutoModelForCausalLM, completions: torch.Tensor):
    """
    Produce tensor of logprobs for each completion, shape (batch, num_rollouts, 
    max_completion_length - 1). There are no log probs for the first token.
    """
    # Get logits over completions
    logits = model(completions).logits
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