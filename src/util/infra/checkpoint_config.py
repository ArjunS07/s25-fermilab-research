"""Lightweight checkpoint architecture reconstruction shared by inference and tests."""

# Architecture keys reconstructed from a checkpoint's embedded full_config. Only the
# fields the mass-shell GNN actually consumes; extra keys in older checkpoints are ignored.
ARCH_KEYS = (
    "n_hidden", "n_layers", "regulator_mass",
    "use_reference_vectors", "include_mass_condition",
)

def resolve_architecture(namespace, checkpoint):
    if not isinstance(checkpoint, dict) or not checkpoint.get("full_config"):
        return namespace, {}
    model = checkpoint["full_config"].get("model", {})
    mismatches = {key: (getattr(namespace, key, None), model[key])
                  for key in ARCH_KEYS
                  if key in model and getattr(namespace, key, None) != model[key]}
    for key in ARCH_KEYS:
        if key in model:
            setattr(namespace, key, model[key])
    return namespace, mismatches
