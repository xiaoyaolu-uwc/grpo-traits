import pytest
import torch
import src.grpo_traits.core as core
import statistics as stats

# =====================================================================================
# 1. Tests for compute_advantage (with std_correct=True)
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
    print(torch.sum(core.compute_advantage(rewards)).cpu())
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
    print(correct)
    print(computed)
    for i in range(len(rewards)):
        assert correct[i] == computed[i]
    
# =====================================================================================
# 2. Tests for compute loss
# =====================================================================================


