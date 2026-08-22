import pickle
import torch

NUM_CLASSES = 5  # Number of jet types
JET_TYPE_TO_INDEX = {name: i for i, name in enumerate(("g", "q", "t", "w", "z"))}
MAX_N_PARTICLES = 150  # Maximum number of particles in a jet
MIN_N_PARTICLES = 4  # Minimum number of particles in a jet

def load_model(model_path, device=torch.device("cpu")):
    model = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(model, dict) and model.get("format") == "jet_attribute_v3_state_dict":
        from models.stage1.jet_attr_model_v3 import JetAttributeFlowV3
        model_v3 = JetAttributeFlowV3(**model["config"])
        model_v3.load_state_dict(model["state_dict"])
        return model_v3.to(device)
    return model

def one_hot_enc_jet_type(y, num_classes=NUM_CLASSES):
    """
    One-hot encode the jet type labels.

    In the order ["g", "q", "t", "w", "z"],
    
    Args:
        y (torch.Tensor): Tensor of jet type labels.
        num_classes (int): Number of classes for one-hot encoding.
        
    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    one_hot_enc =  torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    return one_hot_enc

def global_jet_type_indices(jet_types):
    """Return the fixed five-class indices for configured JetNet type names."""
    unknown = sorted(set(jet_types) - JET_TYPE_TO_INDEX.keys())
    if unknown:
        raise ValueError(f"unknown jet types: {unknown}")
    if not jet_types:
        raise ValueError("jet_types must not be empty")
    return [JET_TYPE_TO_INDEX[jet_type] for jet_type in jet_types]


def local_jet_type_indices(global_indices, jet_types):
    """Map fixed five-class labels into the configured local class order.

    Metrics and report panels use local labels (``0..len(jet_types)-1``), while
    the Stage-1 and Stage-2 conditioning vectors always use the fixed global
    five-class convention.  Keeping the two spaces explicit prevents a q-only
    or t-only run from silently being sampled as a gluon run.
    """
    lookup = torch.full((NUM_CLASSES,), -1, dtype=torch.long,
                        device=global_indices.device)
    for local_index, global_index in enumerate(global_jet_type_indices(jet_types)):
        lookup[global_index] = local_index
    local_indices = lookup[global_indices.long()]
    if (local_indices < 0).any():
        unexpected = torch.unique(global_indices[local_indices < 0]).tolist()
        raise ValueError(
            f"encountered global jet classes {unexpected} outside configured "
            f"jet_types={jet_types}"
        )
    return local_indices


def generate_jets(model, device, jet_types=None, num_jets=1000, one_hot_types=None,
                  n_jet_types=None):
    """
    Use a pretrained normalizing flow model to generate jets.
    Generates jets with features jet_types (in the form of 5 one-hot encoded slots), eta, p_t, mass, and number of particles.

    Args:
        model: Pretrained normalizing flow model.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        jet_types: Configured type names. These are mapped to the fixed global
            five-class encoding [g, q, t, w, z] before sampling.
        num_jets: Number of jets to generate.
        one_hot_types: Optional one-hot encoded tensor of desired jet types. If None, random types
    """
    if one_hot_types is None:
        if jet_types is None:
            # Compatibility for old callers. New code must pass names because a
            # count alone cannot distinguish q-only/t-only from g-only.
            if n_jet_types is None:
                raise ValueError("jet_types must be provided when one_hot_types is absent")
            global_indices = torch.arange(n_jet_types, device=device)
        else:
            global_indices = torch.tensor(global_jet_type_indices(jet_types), device=device)
        sample_jet_types = global_indices[
            torch.randint(0, len(global_indices), (num_jets,), device=device)
        ]
        one_hot_types = one_hot_enc_jet_type(sample_jet_types)
    jets, jet_logprobs = model.sample(num_jets, context=one_hot_types)
    jets[:, -1] = torch.round(jets[:, -1])
    jets[:, -1] = torch.clamp(jets[:, -1], min=MIN_N_PARTICLES, max=MAX_N_PARTICLES) 
    jets = torch.cat([
        one_hot_types,
        jets,
    ], dim=-1).to(device)  # Concatenate along the last dimension
    return jets, jet_logprobs

def generate_masks(num_particles, max_particles_per_jet, device):
    """
    Generate random masks for the particles in each jet.
    
    Args:
        num_particles (batch_size, ): Number of particles per jet.
        max_n_particles (int): Maximum number of particles in a jet.
        device: Device to generate tensors on

    Returns:
        torch.Tensor: Random masks for the particles in each jet.
    """
    num_particles = num_particles.to(device=device, dtype=torch.int64)
    assert (num_particles <= max_particles_per_jet).all()
    positions = torch.arange(max_particles_per_jet, device=device)
    return (positions.unsqueeze(0) < num_particles.unsqueeze(1)).to(torch.float32)
