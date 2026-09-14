"""
Performs the core computations in GRPO which do not require loading model weights, including:
* The completion mask given .... Output of form (n_rollouts, seq_len)
* The advantage given rewards for each rollout. Output of form (n_rollouts)
* The loss for the episode, given advantage & logprobs, both (n_rollouts, seq_len). Output scalar.
"""

import torch
import statistics as stats


def stack_rollouts(rollouts: list, prompt_length: int, 
                   max_length: int, pad_id: int) -> tuple:
    """
    Builds rollouts tensor and completion mask.
    """
    mask = []
    stacked_rollouts = []
    for rollout in rollouts:
        # Compute relevant lengths
        seq_len = rollout.shape[0] - prompt_length
        pad_len = max_length - rollout.shape[0]
        # Build padded rollouts
        padding = torch.full((pad_len, ), pad_id)
        padded = torch.cat((rollout, padding), dim=0)
        stacked_rollouts.append(padded)
        # Build mask
        prompt_indic = torch.full((prompt_length, ), 0)
        completion_indic = torch.full((seq_len, ), 1)
        padding_indic = torch.full((pad_len, ), 0)
        completion_mask = torch.cat([prompt_indic, completion_indic, padding_indic], dim=0)
        mask.append(completion_mask)
    mask = torch.stack(mask)
    stacked_rollouts = torch.stack(stacked_rollouts)
    return stacked_rollouts, mask

def compute_advantage(rewards: list, std_correct: bool = True) -> torch.Tensor:
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
    advantages = torch.tensor(advantages, dtype=torch.float32, device="mps")
    return advantages

def compute_loss(advantages: torch.Tensor, log_probs: torch.Tensor, completion_mask: torch.Tensor,
         max_length: int, aggregation: str = "max_length") -> torch.Tensor:
    """
    Computes loss from advantages and log probabilities, supporting
    different methods for loss aggregation.
    """
    # Check if loss aggregation mode is valid
    if aggregation != "max_length" or "sequence" or "token":
        raise ValueError("aggregation should equal one of 'sequence', 'token', or 'length'")
    # Compute and return loss
    if aggregation == "max_length":
        loss = torch.sum(advantages.detach() * log_probs / max_length)
    elif aggregation == "sequence":
        seq_lengths = torch.sum(completion_mask, axis=-1)[:, None]
        loss = torch.sum(advantages.detach() * log_probs / seq_lengths)
    else:
        num_tokens = torch.sum(completion_mask).item()
        loss = torch.sum(advantage.detach() * log_probs / num_tokens)
    return loss

