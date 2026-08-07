from argparse import Namespace
import torch

from config import generation_controls_from_namespace
from util.geometry.conditioning import scale_condition_pt


def test_generation_controls_preserve_prior_geometry_and_guidance():
    args = Namespace(
        use_cfg=True,
        cfg_guidance_weight=0.5,
        use_hyperbolic=True,
        hyperbolic_model="mass_shell",
        regulator_mass=0.3,
        use_reference_vectors=False,
        sampler="euler",
        prior_dist="axis_aligned",
        mass_shell_max_step_rapidity=0.5,
        mass_shell_max_substeps=64,
        integration_end_time=0.99,
    )

    assert generation_controls_from_namespace(args) == {
        "use_cfg": True,
        "cfg_guidance_weight": 0.5,
        "use_hyperbolic": True,
        "hyperbolic_model": "mass_shell",
        "regulator_mass": 0.3,
        "use_reference_vectors": False,
        "sampler": "euler",
        "prior_dist": "axis_aligned",
        "mass_shell_max_step_rapidity": 0.5,
        "mass_shell_max_substeps": 64,
        "integration_end_time": 0.99,
    }


def test_scaled_pt_condition_contract():
    pt = torch.tensor([100.0, 250.0])
    expected = torch.tensor([0.5, 1.25])
    assert torch.equal(scale_condition_pt(pt, 200.0), expected)
