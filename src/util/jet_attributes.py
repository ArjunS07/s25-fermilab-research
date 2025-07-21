import pickle
import torch

NUM_CLASSES = 5  # Number of jet types
MODEL_PATH = "gen/models/jet_attr_nf_model.pkl"

def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def one_hot_enc_jet_type(y, num_classes=NUM_CLASSES):
    """
    One-hot encode the jet type labels.
    
    Args:
        y (torch.Tensor): Tensor of jet type labels.
        num_classes (int): Number of classes for one-hot encoding.
        
    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    one_hot_enc =  torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    return one_hot_enc

def generate_jets(model, device, n_jet_types, num_jets=1000, one_hot_types=None):
    """
    Use a pretrained normalizing flow model to generate jets.
    Generates jets with features eta, p_t, mass, and number of particles.

    Args:
        model: Pretrained normalizing flow model.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        n_jet_types: Number of jet types for one-hot encoding.
        num_jets: Number of jets to generate.
        one_hot_types: Optional one-hot encoded tensor of desired jet types. If None, random types
    """
    if not one_hot_types:
        sample_jet_types = torch.randint(0, n_jet_types, (num_jets,)).to(device)
        one_hot_types = one_hot_enc_jet_type(sample_jet_types)
    jets, jet_logprobs = model.sample(num_jets, context=one_hot_types)
    jets = torch.cat([
        one_hot_types,  # Add one-hot encoded jet types
        jets,
    ], dim=-1).to(device)  # Concatenate along the last dimension
    return jets, jet_logprobs