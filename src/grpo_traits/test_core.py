import pytest
import torch
from grpo_traits import core

# =====================================================================================
# 1. Tests for stack rollouts
# =====================================================================================
stack_rollouts_data = [
    (torch.tensor([[1, 1, 1, 151643]]), 2, 151643, torch.tensor([0, 1, 1])),
    (torch.tensor([
        [151643, 151643, 1, 2, 2, 151643], 
        [1, 1, 1, 2, 151643, 151643]
                   ]), 3, 151643,
     torch.tensor([
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 0]
         ])),
]

@pytest.mark.parametrize("completions, max_prompt_length, pad_id, mask", stack_rollouts_data)
def test_completion_mask(completions, max_prompt_length, pad_id, mask):
    computed_mask = core.completion_mask(completions, max_prompt_length, pad_id)
    assert torch.all(computed_mask  == mask)

# =====================================================================================
# 2. Tests for compute_advantage (with std_correct=True)
# =====================================================================================
adv_sums_data = [
    ([1, 1, 1, 1, 0, 0, 0, 0], pytest.approx(0, abs=1e-5)),    
    ([1, 1, 0, 0, 0, 0, 0, 0], pytest.approx(0, abs=1e-5)),
    ([1, 0, 1, 0, 1, 0], pytest.approx(0, abs=1e-5)),
    ([-1, 2, 3, 4, 7, 19], pytest.approx(0, abs=1e-5)),
]
adv_uniform_data = [
    ([0, 0, 0, 0], pytest.approx(0, abs=1e-5)),
    ([1, 1, 1, 1], pytest.approx(0, abs=1e-5)),
    ([10, 10, 10, 10, 10, 10], pytest.approx(0, abs=1e-5)),
]

# Advantages add to zero
@pytest.mark.parametrize("rewards,exp_sum", adv_sums_data)
def test_adv_sum(rewards, exp_sum):
    assert torch.sum(core.compute_advantage(rewards)).cpu() == exp_sum

@pytest.mark.parametrize("rewards,exp_val", adv_uniform_data)
# Uniform reward returns zero advantage
def test_adv_uniform(rewards, exp_val):
    for a in core.compute_advantage(rewards).cpu():
        assert a == exp_val

# Matches manual computations
def test_adv_values():
    rewards = [1, 1, 1, 1, 0, 0, 0, 0]
    adv_1 = pytest.approx(0.5 / (0.5 + 1e-5), abs=1e-5)
    adv_0 = pytest.approx(-0.5 / (0.5 + 1e-5), abs=1e-5)
    correct = [adv_1, adv_1, adv_1, adv_1, adv_0, adv_0, adv_0, adv_0]
    computed = core.compute_advantage(rewards).cpu()
    for i in range(len(rewards)):
        assert correct[i] == computed[i]
    
# =====================================================================================
# 3. Tests for compute loss
# =====================================================================================

def test_loss():
    advantages = torch.tensor([0.7071, 0.7071, -1.4142])[:, None]
    log_probs = torch.tensor([
        [-3, -2, -2, -8],
        [-3, -1, -8, -8],
        [-3, -4, -2, -2]
    ])
    completion_mask = torch.tensor([
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 1]
    ])
    max_length = 4
    expected_loss = pytest.approx(-1.944525, rel=1e-5)
    assert core.compute_loss(advantages, log_probs, completion_mask, max_length) == expected_loss
