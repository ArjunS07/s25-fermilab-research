import torch

from config import InferRunConfig, generation_controls_from_config
from util.geometry.conditioning import scale_condition_pt


def test_generation_controls_from_config():
    cfg = InferRunConfig(
        inference={
            "use_cfg": True, "cfg_guidance_weight": 0.5,
            "integration_end_time": 0.99,
        },
        model={"regulator_mass": 0.3},
    )
    assert generation_controls_from_config(cfg, "axis_aligned") == {
        "use_cfg": True,
        "cfg_guidance_weight": 0.5,
        "regulator_mass": 0.3,
        "use_reference_vectors": True,
        "include_mass_condition": True,
        "use_hyperbolic": True,
        "integration_end_time": 0.99,
        "prior_dist": "axis_aligned",
    }


def test_euclidean_generation_controls_disable_shell_integration():
    cfg = InferRunConfig(model={
        "architecture": "lorentznet",
        "flow_geometry": "euclidean",
        "reference_mode": "none",
        "use_reference_vectors": False,
    })
    controls = generation_controls_from_config(cfg, "axis_aligned_per_jet")
    assert controls["use_hyperbolic"] is False
    assert controls["use_reference_vectors"] is False


def test_scaled_pt_condition_contract():
    pt = torch.tensor([100.0, 250.0])
    expected = torch.tensor([0.5, 1.25])
    assert torch.equal(scale_condition_pt(pt, 200.0), expected)
