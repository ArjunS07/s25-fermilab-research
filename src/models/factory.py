"""Construction dispatch for the frozen type-A model and LorentzNet ablation."""

from models.lorentznet_flow import build_lorentznet
from models.mass_shell_gnn import LEFTJeN


def build_flow_model(model_config, max_num_jet_types: int):
    common = dict(
        max_num_jet_types=max_num_jet_types,
        num_layers=model_config.n_layers,
        hidden_dim=model_config.n_hidden,
        include_pt=True,
        include_mass_condition=model_config.include_mass_condition,
        regulator_mass=model_config.regulator_mass,
    )
    if model_config.architecture == "mass_shell_gnn":
        return LEFTJeN(
            **common,
            use_reference_vectors=model_config.use_reference_vectors,
        )
    if model_config.architecture == "lorentznet":
        return build_lorentznet(
            **common,
            flow_geometry=model_config.flow_geometry,
            reference_mode=model_config.reference_mode,
            scalar_init_mode=model_config.scalar_init_mode,
        )
    raise ValueError(f"unknown model architecture: {model_config.architecture!r}")


__all__ = ["build_flow_model"]
