import copy

import pytest
import torch

from cr_rl.training.resume_options import override_resume_rates


def test_resume_rate_transition_keeps_adam_steps_and_moments():
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    a = torch.optim.AdamW(actor.parameters(), lr=2e-5)
    c = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    actor(torch.ones(1, 2)).sum().backward()
    a.step()
    before = copy.deepcopy(a.state_dict())
    report = override_resume_rates(a, c, actor_lr=1e-5)
    assert report["actor"] == dict(before=[2e-5], after=[1e-5], explicit_override=True)
    assert report["critic"] == dict(before=[1e-4], after=[1e-4], explicit_override=False)
    after = a.state_dict()
    for key, state in before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(after["state"][key][name], value, atol=0, rtol=0)
    assert override_resume_rates(a, c)["actor"]["after"] == [1e-5]
    with pytest.raises(ValueError, match="finite"):
        override_resume_rates(a, c, actor_lr=1e-6, critic_lr=float("nan"))
    assert a.param_groups[0]["lr"] == 1e-5
