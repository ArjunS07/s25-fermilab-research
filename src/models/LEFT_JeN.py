"""Compatibility constructor for current and historical LEFTJeN checkpoints.

New code should import :class:`MassShellFlow` directly. ``LEFTJeN`` remains as a
small factory so archived configs and checkpoint-analysis tools keep working.
"""

from models.legacy_leftjen import LegacyLEFTJeN, masked_neighbor_softmax, psi
from models.mass_shell_flow import MassShellFlow


def LEFTJeN(max_num_jet_types, max_particles=150, embed_dim=128,
            num_layers=6, message_dim=128, hidden_dim=128,
            use_residual_update=True, include_pt=False,
            use_reference_vectors=False, use_node_scalars=False,
            node_scalar_seed="physics", node_scalar_dim=None,
            use_adaln=False, use_attention=False, backbone="legacy",
            include_mass_condition=False, num_attention_heads=8,
            vector_channels=16, regulator_mass=0.5,
            velocity_readout_init="small_normal"):
    if backbone == "tangent_attention":
        if not use_reference_vectors:
            raise ValueError("tangent_attention requires typed reference vectors")
        condition_dim = (
            max_num_jet_types + 1 + int(include_pt) + int(include_mass_condition)
        )
        return MassShellFlow(
            condition_dim=condition_dim,
            n_particle_types=max_num_jet_types,
            width=hidden_dim,
            num_layers=num_layers,
            num_heads=num_attention_heads,
            vector_channels=vector_channels,
            regulator_mass=regulator_mass,
            readout_init=velocity_readout_init,
        )
    if backbone != "legacy":
        raise ValueError(f"unknown backbone {backbone!r}")
    return LegacyLEFTJeN(
        max_num_jet_types=max_num_jet_types,
        max_particles=max_particles,
        embed_dim=embed_dim,
        num_layers=num_layers,
        message_dim=message_dim,
        hidden_dim=hidden_dim,
        use_residual_update=use_residual_update,
        include_pt=include_pt,
        use_reference_vectors=use_reference_vectors,
        use_node_scalars=use_node_scalars,
        node_scalar_seed=node_scalar_seed,
        node_scalar_dim=node_scalar_dim,
        use_adaln=use_adaln,
        use_attention=use_attention,
        backbone="legacy",
        include_mass_condition=include_mass_condition,
    )


__all__ = ["LEFTJeN", "LegacyLEFTJeN", "MassShellFlow", "masked_neighbor_softmax", "psi"]
