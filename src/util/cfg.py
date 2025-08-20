import torch
def null_vector(n_jet_types):
    """
    Constant null vector to use when dropping out jet information.
    Jet type one-hot encoded embedding: (0, 0, 0, 0, 0), # constituents: -1
    """
    zero_hot_jet_type = torch.zeros(n_jet_types)
    constituents = torch.tensor([0])
    return torch.cat([zero_hot_jet_type, constituents])

def null_vector_like(jet_info, max_n_jet_types=5):
    """
    Create a null vector with the same shape as the jet information tensor.
    """
    batch_size = jet_info.shape[0]
    return null_vector(max_n_jet_types).unsqueeze(0).repeat(batch_size, 1)