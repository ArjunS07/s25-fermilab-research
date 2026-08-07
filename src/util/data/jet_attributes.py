import pickle
import torch

NUM_CLASSES = 5  # Number of jet types
MAX_N_PARTICLES = 150  # Maximum number of particles in a jet
MIN_N_PARTICLES = 4  # Minimum number of particles in a jet

def load_model(model_path, device=torch.device("cpu")):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except AttributeError as error:
        # One pre-publication Stage-1 v2 artifact was serialized while its training script
        # was __main__. Recover that exact checkpoint without weakening the portable format
        # used by all subsequent saves.
        if "JetAttributeModelV2" not in str(error):
            raise
        import __main__
        from models.stage1.jet_attr_model_v2 import JetAttributeModelV2
        previous = getattr(__main__, "JetAttributeModelV2", None)
        __main__.JetAttributeModelV2 = JetAttributeModelV2
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        finally:
            if previous is None:
                delattr(__main__, "JetAttributeModelV2")
            else:
                __main__.JetAttributeModelV2 = previous
    if isinstance(model, dict) and model.get("format") == "jet_attribute_v2_state_dict":
        from models.stage1.jet_attr_model_v2 import JetAttributeModelV2
        model_v2 = JetAttributeModelV2(**model["config"])
        model_v2.load_state_dict(model["state_dict"])
        return model_v2.to(device)
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

def one_hot_to_type(one_hot_types):
    """
    Convert one-hot encoded jet types back to class labels.
    
    Args:
        one_hot_types (torch.Tensor): One-hot encoded tensor of jet types.
        
    Returns:
        torch.Tensor: Class labels for the jet types.
    """
    return torch.argmax(one_hot_types, dim=1)

def generate_jets(model, device, n_jet_types, num_jets=1000, one_hot_types=None):
    """
    Use a pretrained normalizing flow model to generate jets.
    Generates jets with features jet_types (in the form of 5 one-hot encoded slots), eta, p_t, mass, and number of particles.

    Args:
        model: Pretrained normalizing flow model.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        n_jet_types: Desired number of jet types for one-hot encoding (<= 5)
        num_jets: Number of jets to generate.
        one_hot_types: Optional one-hot encoded tensor of desired jet types. If None, random types
    """
    if one_hot_types is None:
        sample_jet_types = torch.randint(0, n_jet_types, (num_jets,)).to(device)
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
    num_particles = num_particles.to(torch.int64)
    assert (num_particles <= max_particles_per_jet).all()
    masks = [
        torch.concat((
            torch.ones(int(n_ones), device=device),
            torch.zeros(max_particles_per_jet - int(n_ones), device=device)
        ))
        for n_ones in num_particles
    ]
    return torch.stack(masks)
