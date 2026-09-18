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
    truncate_mask = mask[:, 1:].int()
    return truncate_mask

def compute_advantage(rewards: list, group_size: int, std_correct: bool = True) -> torch.Tensor:
    """
    Computes advantages from rewards, optionally without division by std.
    """
    num_batches = int(len(rewards) / group_size)
    advantages = []
    for i in range(num_batches):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        batch_rewards = rewards[start_idx:end_idx]
        mean_r = stats.mean(batch_rewards)
        std_r = stats.pstdev(batch_rewards)
        if std_correct:
            advantages += [(r - mean_r) /  (std_r + 1e-5) for r in batch_rewards]
        else:
            advantages += [(r - mean_r) for r in batch_rewards]
    advantages = torch.tensor(advantages, dtype=torch.float32, )[:, None]
    return advantages

def compute_loss(advantages: torch.Tensor, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                 clip_eps: float, completion_mask: torch.Tensor,
         max_new: int, aggregation: str = "max_length") -> torch.Tensor:
    """
    Computes loss from advantages and log probabilities, supporting
    different methods for loss aggregation.
    """
    # Check if loss aggregation mode is valid
    valid_aggregations = ["max_length", "sequence", "token"]
    if aggregation not in valid_aggregations:
        raise ValueError("aggregation should equal one of 'sequence', 'token', or 'max_length'")

    # Compute probability ratio between new and old probs
    ratios = torch.exp(log_probs - old_log_probs) 
    A = advantages.detach()
    surrogate_loss = torch.minimum(ratios * A, torch.clamp(
        ratios, 1 - clip_eps, 1 + clip_eps) * A) * completion_mask

    # Compute and return loss
    if aggregation == "max_length":
        loss = - torch.sum(surrogate_loss) / max_new
    elif aggregation == "sequence":
        seq_lengths = torch.sum(completion_mask, axis=-1)[:, None]
        loss = - torch.sum(surrogate_loss / seq_lengths)
    else:
        num_tokens = torch.sum(completion_mask).item()
        loss = - torch.sum(surrogate_loss) / num_tokens
    return loss

