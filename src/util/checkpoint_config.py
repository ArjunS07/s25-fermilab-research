"""Lightweight checkpoint architecture reconstruction shared by inference and tests."""

ARCH_KEYS = (
    "n_hidden", "n_layers", "use_residual", "use_reference_vectors",
    "use_node_scalars", "node_scalar_seed", "use_adaln", "use_attention",
    "use_hyperbolic", "hyperbolic_model", "regulator_mass", "backbone",
    "include_mass_condition", "num_attention_heads", "vector_channels",
    "velocity_readout_init", "geometric_state", "use_global_pooling",
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
