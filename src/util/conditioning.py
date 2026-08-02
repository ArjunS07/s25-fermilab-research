"""Shared Stage-2 conditioning conventions."""

def scale_condition_pt(gen_pt, final_scale, backbone):
    """Geometric backbones consume model-scaled pT; legacy consumes raw pT."""
    return (gen_pt / final_scale
            if backbone in ("tangent_attention", "mass_shell_gnn") else gen_pt)
