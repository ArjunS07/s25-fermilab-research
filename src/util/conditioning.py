"""Shared Stage-2 conditioning conventions."""

def scale_condition_pt(gen_pt, final_scale):
    """The mass-shell GNN consumes model-scaled pT."""
    return gen_pt / final_scale
