import argparse
import pickle
import os

import numpy as np
import torch
import normflows as nf

from data import get_data_path
from util.jet_attributes import one_hot_enc_jet_type, NUM_CLASSES

RANDOM_SEED = 42


def get_model_pth_path(output_path):
    return os.path.join(output_path, "jet_attr_model.pth")

def noise_num_particles(X, noise_std=0.15):
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

    parser = argparse.ArgumentParser(description="Train Jet Attribute NF Model on JetNet dataset")
    parser.add_argument("--output_path", type=str, default="/mnt/data/output", help="Path to save the output files")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=35_000, help="Number of epochs to train the model")
    parser.add_argument("--K", type=int, default=10, help="Number of layers in the flow")
    parser.add_argument("--hidden_units", type=int, default=128, help="Number of hidden units in the flow")
    parser.add_argument("--hidden_layers", type=int, default=8, help="Number of hidden layers in the flow")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the trained model")

    args = parser.parse_args()
    data_path = get_data_path(args.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    # Only take global jet features
    X_train = X_train[:][1]

    # equalize the number of jets per type
    X_train_equalized = []
    jet_types_ints = set(X_train[:, -1].long().tolist())
    min_jets_per_type = min(len(X_train[X_train[:, -1] == jet_type]) for jet_type in jet_types_ints)
    for jet_type in jet_types_ints:
        X_train_equalized.extend(X_train[X_train[:, -1] == jet_type][:min_jets_per_type])
    X_train = torch.tensor(np.array(X_train_equalized), dtype=torch.float32)


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

    K = args.K
    hidden_units = args.hidden_units
    hidden_layers = args.hidden_layers

    # Fixed
    latent_size = 4
    context_size = NUM_CLASSES

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    model = nf.ConditionalNormalizingFlow(q0, flows).to(device)

    torch.manual_seed(RANDOM_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    epochs = args.num_epochs
    batch_size = args.batch_size

    loss_hist = np.array([])
    for it in range(epochs):
        optimizer.zero_grad()

        indices = torch.randperm(len(X_train), device=device)[:batch_size]
        jets = X_train[indices]
        jet_info = jets[:, :-1]  # Exclude the jet type column
        jet_types = jets[:, -1].long()  # Get the jet type column
        one_hot_types = one_hot_enc_jet_type(jet_types).to(device)
        
        loss = model.forward_kld(jet_info, one_hot_types)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    # save loss as csv
    np.savetxt(f"{args.output_path}/jet_attr_model_loss_hist.csv", loss_hist, delimiter=",")
    if args.save_model:
        with open(get_model_pth_path(args.output_path), "wb") as f:
            torch.save(model, f)

    with open(f"{args.output_path}/jet_attr_model_info.txt", "w") as f:
        f.write(f"CLI args:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.close()