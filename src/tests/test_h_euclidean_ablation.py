from pathlib import Path

import torch

from config import TrainRunConfig, build_config
from models.factory import build_flow_model
from training import flow_matching_loss
from util.geometry.euclidean import euclidean_ode_step


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _config(name, overrides=None):
    return build_config(TrainRunConfig, str(CONFIG_DIR / name), overrides or [])


def test_h_euclidean_331k_is_a_controlled_geometry_stack_ablation():
    rfm = _config(
        "g30-lorentznet-h-rfm-logmap-tangentrefs-994k.yaml",
        [
            "training.num_epochs=800",
            "training.max_optimizer_steps=331000",
            "training.stability_probe_steps=[0,1000,5000,20000,40000,80000,150000,200000,331000]",
        ],
    )
    euclidean = _config("g30-lorentznet-h-euclidean-331k.yaml")

    # Data, optimization, coupling, prior, inference, conditioning, and RNG streams
    # are frozen.  In particular, ICP remains online geodesic ICP in both arms.
    assert euclidean.data == rfm.data
    assert euclidean.training == rfm.training
    assert euclidean.inference == rfm.inference
    assert euclidean.paths == rfm.paths
    assert euclidean.training.coupling == "online_geodesic_icp"

    rfm_model = rfm.model.model_dump()
    euclidean_model = euclidean.model.model_dump()
    differing = {
        key for key in rfm_model if rfm_model[key] != euclidean_model[key]
    }
    assert differing == {
        "flow_geometry",
        "reference_mode",
        "particle_readout_mode",
        "field_degree_normalization",
    }
    assert euclidean.model.flow_geometry == "euclidean"
    assert euclidean.model.reference_mode == "plain_readout"
    assert euclidean.model.particle_readout_mode == "ambient"
    assert euclidean.model.field_degree_normalization == "none"


def test_h_euclidean_control_is_parameter_matched_to_rfm_h():
    rfm = _config("g30-lorentznet-h-rfm-logmap-tangentrefs-994k.yaml")
    euclidean = _config("g30-lorentznet-h-euclidean-331k.yaml")
    rfm_model = build_flow_model(rfm.model, max_num_jet_types=5)
    euclidean_model = build_flow_model(euclidean.model, max_num_jet_types=5)

    assert list(rfm_model.state_dict()) == list(euclidean_model.state_dict())
    assert all(
        left.shape == right.shape
        for left, right in zip(
            rfm_model.state_dict().values(), euclidean_model.state_dict().values()
        )
    )
    assert sum(p.numel() for p in rfm_model.parameters()) == sum(
        p.numel() for p in euclidean_model.parameters()
    )


def test_h_euclidean_config_has_finite_training_and_sampling_paths():
    cfg = _config("g30-lorentznet-h-euclidean-331k.yaml")
    model = build_flow_model(cfg.model, max_num_jet_types=5).double()
    generator = torch.Generator().manual_seed(731)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float64
    )
    x0 = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)
    x1 = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)
    x0 *= mask.unsqueeze(-1)
    x1 *= mask.unsqueeze(-1)
    conditions = torch.randn(2, 8, generator=generator, dtype=torch.float64)
    references = torch.randn(2, 2, 4, generator=generator, dtype=torch.float64)
    references[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    t = torch.tensor([0.2, 0.7], dtype=torch.float64)

    loss = flow_matching_loss(
        model=model,
        raw_model=model,
        config=cfg.model,
        x0=x0,
        x1=x1,
        t=t,
        mask=mask,
        conditions=conditions,
        references=references,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    sampled = x0
    times = torch.linspace(0, 1, 5, dtype=torch.float64)
    for start, end in zip(times[:-1], times[1:]):
        sampled = euclidean_ode_step(
            model,
            sampled,
            conditions,
            mask,
            start,
            end,
            references=references,
        )
    assert torch.isfinite(sampled).all()
    assert torch.equal(sampled[mask == 0], torch.zeros_like(sampled[mask == 0]))
