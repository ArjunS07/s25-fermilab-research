"""Constructor for the mass-shell GNN, the single reference architecture.

New code should import :class:`MassShellGNNFlow` directly. ``LEFTJeN`` remains a small
factory so checkpoint-analysis tooling and existing call sites keep working; it accepts
(and ignores) the historical architecture kwargs that older checkpoints embed.
"""

from models.mass_shell_gnn import MassShellGNNFlow


def LEFTJeN(max_num_jet_types, num_layers=6, hidden_dim=128, include_pt=False,
            include_mass_condition=False, regulator_mass=0.5,
            use_reference_vectors=True, **_ignored_legacy):
    if not use_reference_vectors:
        raise ValueError("mass_shell_gnn requires typed reference vectors")
    condition_dim = max_num_jet_types + 1 + int(include_pt) + int(include_mass_condition)
    return MassShellGNNFlow(condition_dim, max_num_jet_types, hidden_dim, num_layers,
                            regulator_mass)


__all__ = ["LEFTJeN", "MassShellGNNFlow"]
