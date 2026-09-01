import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


class _BaseModel(torch.nn.Module):
    pass


def _load_dmd_module(monkeypatch, relative_path):
    pipeline = types.ModuleType("pipeline")
    pipeline.SelfForcingTrainingPipeline = object
    pipeline.RollingForcingTrainingPipeline = object

    model_base = types.ModuleType("model.base")
    model_base.SelfForcingModel = _BaseModel
    model_base.RollingForcingModel = _BaseModel

    wan_wrapper = types.ModuleType("utils.wan_wrapper")

    class _LegacyWanWrapper:
        @staticmethod
        def _convert_x0_to_flow_pred(**kwargs):
            return torch.full_like(kwargs["x0_pred"], -99.0)

    wan_wrapper.WanDiffusionWrapper = _LegacyWanWrapper

    monkeypatch.setitem(sys.modules, "pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "model.base", model_base)
    monkeypatch.setitem(sys.modules, "utils.wan_wrapper", wan_wrapper)

    module_name = "dmd_" + relative_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Scheduler:
    alphas_cumprod = None

    def __init__(self):
        self.converted_x0 = False

    def add_noise(self, original_samples, noise, timestep):
        return original_samples

    def convert_x0_to_noise(self, x0, xt, timestep):
        self.converted_x0 = True
        return x0 + 7.0


def _make_harness(loss_type):
    batch_size, num_frames = 2, 3
    shape = [batch_size, num_frames, 1, 2, 2]
    generated = torch.zeros(shape)
    raw_flow = torch.arange(1, generated.numel() + 1, dtype=torch.float32).reshape(shape)
    raw_flow.requires_grad_()
    predicted_x0 = torch.full(shape, 13.0, requires_grad=True)
    timestep = torch.full((batch_size, num_frames), 200)
    captured = {}

    scheduler = _Scheduler()
    harness = types.SimpleNamespace(
        args=types.SimpleNamespace(denoising_loss_type=loss_type),
        ts_schedule=False,
        ts_schedule_max=False,
        min_score_timestep=0,
        num_train_timestep=1000,
        num_frame_per_block=1,
        timestep_shift=1.0,
        min_step=20,
        max_step=980,
        scheduler=scheduler,
    )
    harness._run_generator = lambda **kwargs: (generated, None, None, None)
    harness._get_timestep = lambda *args, **kwargs: timestep
    harness.fake_score = lambda **kwargs: (raw_flow, predicted_x0)

    def denoising_loss_func(**kwargs):
        captured.update(kwargs)
        prediction = kwargs["flow_pred"]
        if prediction is None:
            prediction = kwargs["noise_pred"]
        return prediction.square().mean()

    harness.denoising_loss_func = denoising_loss_func
    return harness, shape, raw_flow, predicted_x0, captured


def test_flow_critic_uses_the_models_native_flow_prediction(monkeypatch):
    module = _load_dmd_module(monkeypatch, "model/dmd.py")
    harness, shape, raw_flow, _, captured = _make_harness("flow")

    loss, _ = module.DMD.critic_loss(
        harness,
        image_or_video_shape=shape,
        conditional_dict={},
        unconditional_dict={},
        clean_latent=None,
    )

    expected = raw_flow.flatten(0, 1)
    torch.testing.assert_close(captured["flow_pred"], expected)
    assert captured["noise_pred"] is None
    assert not harness.scheduler.converted_x0

    loss.backward()
    assert raw_flow.grad is not None
    assert torch.count_nonzero(raw_flow.grad) == raw_flow.numel()


def test_non_flow_critic_still_converts_the_x0_prediction(monkeypatch):
    module = _load_dmd_module(monkeypatch, "model/dmd.py")
    harness, shape, _, predicted_x0, captured = _make_harness("noise")

    module.DMD.critic_loss(
        harness,
        image_or_video_shape=shape,
        conditional_dict={},
        unconditional_dict={},
        clean_latent=None,
    )

    torch.testing.assert_close(captured["noise_pred"], predicted_x0 + 7.0)
    assert captured["flow_pred"] is None
    assert harness.scheduler.converted_x0
