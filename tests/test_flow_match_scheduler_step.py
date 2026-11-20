import importlib.util
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The root project and long_video/ each ship a byte-identical copy of the
# scheduler, and imports resolve to one or the other depending on the working
# directory. Both must behave the same way.
SCHEDULERS = [
    pytest.param(
        _load(_ROOT / "utils" / "scheduler.py", "root_scheduler"),
        id="root",
    ),
    pytest.param(
        _load(_ROOT / "long_video" / "utils" / "scheduler.py", "long_video_scheduler"),
        id="long_video",
    ),
]


def make_scheduler(module):
    scheduler = module.FlowMatchScheduler(
        num_inference_steps=8, sigma_min=0.0, extra_one_step=True
    )
    scheduler.set_timesteps(8)
    return scheduler


def expected_step(scheduler, timestep_index):
    sigma = float(scheduler.sigmas[timestep_index])
    sigma_next = (
        0.0
        if timestep_index + 1 >= len(scheduler.timesteps)
        else float(scheduler.sigmas[timestep_index + 1])
    )
    return sigma_next - sigma


@pytest.mark.parametrize("module", SCHEDULERS)
def test_step_advances_each_sample_by_its_own_timestep(module):
    scheduler = make_scheduler(module)
    # Two samples in one batch: one in the middle of the schedule, one at its
    # last step. A unit model output makes the update equal to the sigma delta.
    timestep = torch.tensor(
        [float(scheduler.timesteps[3]), float(scheduler.timesteps[7])]
    )
    sample = torch.zeros(2, 1, 1, 1)
    model_output = torch.ones(2, 1, 1, 1)

    prev_sample = scheduler.step(model_output, timestep, sample)

    assert torch.allclose(
        prev_sample.flatten(),
        torch.tensor(
            [expected_step(scheduler, 3), expected_step(scheduler, 7)]
        ),
        atol=1e-6,
    )


@pytest.mark.parametrize("module", SCHEDULERS)
def test_step_final_timestep_uses_zero_sigma(module):
    scheduler = make_scheduler(module)
    timestep = torch.tensor([float(scheduler.timesteps[-1])])
    prev_sample = scheduler.step(
        torch.ones(1, 1, 1, 1), timestep, torch.zeros(1, 1, 1, 1)
    )
    assert torch.allclose(
        prev_sample.flatten(),
        torch.tensor([expected_step(scheduler, len(scheduler.timesteps) - 1)]),
        atol=1e-6,
    )


@pytest.mark.parametrize("module", SCHEDULERS)
def test_step_2d_timesteps_are_per_sample(module):
    scheduler = make_scheduler(module)
    timestep = torch.tensor(
        [
            [float(scheduler.timesteps[2]), float(scheduler.timesteps[6])],
            [float(scheduler.timesteps[7]), float(scheduler.timesteps[0])],
        ]
    )
    model_output = torch.ones(4, 1, 1, 1)
    sample = torch.zeros(4, 1, 1, 1)

    prev_sample = scheduler.step(model_output, timestep, sample)

    assert torch.allclose(
        prev_sample.flatten(),
        torch.tensor([expected_step(scheduler, i) for i in (2, 6, 7, 0)]),
        atol=1e-6,
    )
