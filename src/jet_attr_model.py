import pickle
import os

import numpy as np

import torch
import normflows as nf
from jetnet.datasets import JetNet
from util.jet_attributes import one_hot_enc_jet_type, one_hot_to_type

RANDOM_SEED = 42

jet_type_map = {
0: "Gluons",
1: "Light quarks",
2: "Top quarks",
3: "W bosons",
4: "Z bosons"
}

def noise_num_particles(X, noise_std=0.25):
    """
    Add noise to the number of particles in the jets.
    
    Args:
        X (torch.Tensor): Input tensor with jet features.
        noise_std (float): Standard deviation of the noise to be added.
        
    Returns:
        torch.Tensor: Tensor with noisy number of particles.
    """
    num_particles = X[:, 3]
    noise = torch.randn_like(num_particles) * noise_std
    noise = torch.clamp(noise, -1, 1)
    return num_particles + noise

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)

    with open("data/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    # Only take global jet features
    X_train = X_train[:][1]

    # equalize the number of jets per type
    X_train_equalized = []
    min_jets_per_type = min(len(X_train[X_train[:, -1] == jet_type]) for jet_type in jet_type_map.keys())
    for jet_type in jet_type_map.keys():
        X_train_equalized.extend(X_train[X_train[:, -1] == jet_type][:min_jets_per_type])

    X_train = torch.tensor(np.array(X_train_equalized), dtype=torch.float32)

    os.makedirs("gen", exist_ok=True)
    os.makedirs("gen/figs", exist_ok=True)
    os.makedirs("gen/logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_bins = 200
    std = 0.15
    num_particles = X_train[:, 3]
    noised_particles = noise_num_particles(X_train, noise_std=std)
    X_train[:, 3] = noised_particles

    X_train = X_train.to(device)

    long_types = X_train[:, -1].long()
    one_hot_jets = one_hot_enc_jet_type(long_types)
    print(one_hot_jets.shape, one_hot_jets.sum(dim=0))

    K = 6

    latent_size = 4
    context_size = 5
    hidden_units = 128
    hidden_layers = 8

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    model = nf.ConditionalNormalizingFlow(q0, flows).to(device)

    torch.manual_seed(RANDOM_SEED)
    max_iter = 35_000
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    batch_size = 8192

    loss_hist = np.array([])
    for it in range(max_iter):
        optimizer.zero_grad()

        indices = torch.randperm(len(X_train), device=device)[:batch_size]
        jets = X_train[indices]
        jet_info = jets[:, :-1]  # Exclude the jet type column
        jet_types = jets[:, -1].long()  # Get the jet type column
        one_hot_types = one_hot_enc_jet_type(jet_types).to(device)
        
        # Compute loss
        loss = model.forward_kld(jet_info, one_hot_types)

        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    with open("upload/jet_attr_nf_model.pkl", "wb") as f:
        pickle.dump(model, f)