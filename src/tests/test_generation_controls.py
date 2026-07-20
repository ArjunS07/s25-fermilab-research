from argparse import Namespace

from config import generation_controls_from_namespace


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
    }
