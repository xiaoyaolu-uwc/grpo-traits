"""
Performs the core computations in GRPO which do not require loading model weights, including:
* The completion mask given .... Output of form (n_rollouts, seq_len)
* The advantage given rewards for each rollout. Output of form (n_rollouts, 1)
* The loss for the episode, given advantage (n_rollouts) & logprobs (n_rollouts, seq_len). Output scalar.
"""

import torch
import statistics as stats


def completion_mask(completions: torch.Tensor, max_prompt_length: int, 
                    pad_id: int) -> torch.Tensor:
    """
    Generate completion mask given completions.
    """
    is_pad = completions == pad_id
    is_pad[:, :max_prompt_length] = False # Ignore the padding at the front 
    cumsum = torch.cumsum(is_pad.int(), dim=-1)
    mask = (cumsum - is_pad.int()) == 0 
    mask[:, :max_prompt_length] = False
    return mask.int()

def compute_advantage(rewards: list, std_correct: bool = True, device: str = None) -> torch.Tensor:
    """
    Computes advantages from rewards, optionally without division by std.
    """
    mean_r = stats.mean(rewards)
    print(mean_r)
    std_r = stats.pstdev(rewards)
    print(std_r)
    if std_correct:
        advantages = [(r - mean_r) /  (std_r + 1e-5) for r in rewards]
    else:
        advantages = [(r - mean_r) for r in rewards]
    advantages = torch.tensor(advantages, dtype=torch.float32, )[:, None]
    return advantages

def compute_loss(advantages: torch.Tensor, log_probs: torch.Tensor, completion_mask: torch.Tensor,
         max_new: int, aggregation: str = "max_length") -> torch.Tensor:
    """
    Computes loss from advantages and log probabilities, supporting
    different methods for loss aggregation.
    """
    # Check if loss aggregation mode is valid
    valid_aggregations = ["max_length", "sequence", "token"]
    if aggregation not in valid_aggregations:
        raise ValueError("aggregation should equal one of 'sequence', 'token', or 'max_length'")
    num_rollouts = advantages.shape[0]
    # Compute and return loss
    if aggregation == "max_length":
        loss = - torch.sum(advantages.detach() * log_probs * completion_mask) / max_new
    elif aggregation == "sequence":
        seq_lengths = torch.sum(completion_mask, axis=-1)[:, None]
        loss = - torch.sum(advantages.detach() * log_probs * completion_mask / seq_lengths)
    else:
        num_tokens = torch.sum(completion_mask).item()
        loss = - torch.sum(advantages.detach() * log_probs * completion_mask) / num_tokens
    return loss

