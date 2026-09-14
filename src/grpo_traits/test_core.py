import pytest
import torch
import src.grpo_traits.core as core
import statistics as stats

# =====================================================================================
# 1. Tests for stack rollouts
# =====================================================================================
stack_rollouts_data = [
    ([torch.tensor([1, 1, 1])], 1, 4, 151643, 
     torch.tensor([1, 1, 1, 151643]), torch.tensor([0, 1, 1, 0])),
    ([torch.tensor([1, 1, 2, 2]), torch.tensor([1, 1])], 1, 4, 151643,
     # stacked padded tensors
     torch.tensor([[1, 1, 2, 2], 
                   [1, 1, 151643, 151643]]),
    # completion mask
     torch.tensor([[0, 1, 1, 1],
                  [0, 1, 0, 0]])),
]
@pytest.mark.parametrize("rollouts,prompt_length,max_length,pad_id,stacked,mask", stack_rollouts_data)
def test_stack_rollouts(rollouts, prompt_length, max_length, pad_id, stacked, mask):
    computed_stacked, computed_mask = core.stack_rollouts(rollouts, prompt_length, max_length, pad_id)
    print(computed_stacked)
    print(computed_mask)
    assert torch.all(computed_stacked == stacked)
    assert torch.all(computed_mask == mask)

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
# 2. Tests for compute loss
# =====================================================================================


