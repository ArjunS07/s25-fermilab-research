"""Stage-1 jet-attribute model (categorical multiplicity + conditional spline flow, "v3")."""
import os


def get_model_pth_path(output_path):
    """Canonical on-disk path for the trained Stage-1 jet-attribute model."""
    return os.path.join(output_path, "jet_attr_model.pth")
