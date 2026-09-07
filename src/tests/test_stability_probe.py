import json
from types import SimpleNamespace

import torch

import training.stability_probe as stability_probe


def test_nonfinite_probe_is_recorded_without_aborting_training(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 2).double()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    jet_attr_model = object()
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            stability_probe_save_checkpoints=False,
            batch_size=4,
            prior_dist="axis_aligned_lognormal",
        ),
        inference=SimpleNamespace(
            seed=42,
            stability_probe_integration_steps=8,
            stability_probe_samples=16,
            use_cfg=False,
            cfg_guidance_weight=1.0,
            integration_end_time=0.99999,
        ),
        data=SimpleNamespace(num_particles=30, jet_types=["g"]),
        model=SimpleNamespace(regulator_mass=0.1),
        paths=SimpleNamespace(output_path=str(tmp_path)),
    )

    def fail_generation(**_kwargs):
        raise FloatingPointError("mass-shell Euler step produced a non-finite state")

    monkeypatch.setattr(stability_probe, "generate_samples", fail_generation)

    returned = stability_probe.run_stability_probe(
        optimizer_step=0,
        epoch=-1,
        minibatch=None,
        probe_steps={0},
        model=model,
        optimizer=optimizer,
        scheduler=None,
        ema=None,
        losses=[],
        cfg=cfg,
        device=torch.device("cpu"),
        model_output_path=str(tmp_path),
        train_loader=None,
        run_config={},
        full_config={},
        schedule_definition={},
        final_scale=1.0,
        jet_attr_model=jet_attr_model,
    )

    assert returned is jet_attr_model
    assert model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, initial_state[key])
    summary_path = tmp_path / "stability_probes" / "step_000000" / "probe_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["n_unstable"] == 16
    assert summary["failure_reason_counts"] == {"floating_point_error": 16}
    assert "non-finite state" in summary["error"]
