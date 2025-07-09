import os
import argparse
import pickle

import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import anderson

import matplotlib.pyplot as plt
import seaborn as sns

from jetnet.datasets import JetNet
import jetnet.evaluation as eval
from jetnet.utils import cartesian_to_EtaPhiPtE

from FMLorentzNet import LorentzFMNet, ode_solver_methods
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.file_management import make_clear_folder

RANDOM_SEED = 42

MASK = True
NUM_PARTICLES = 30
TRAIN_SPLIT = 0.7

feature_maxes = JetNet.fpnd_norm.feature_maxes
if MASK:
    feature_maxes = feature_maxes + [1]

data_args = {
    # "jet_type": ["g", "q", "t"],
    "jet_type": ["g"],
    "data_dir": "datasets/jetnet",
    "num_particles": NUM_PARTICLES,
    "particle_features": (
        JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1]
    ),
    # The order of the list is preserved in the retrieved data
    "jet_features": ["eta", "pt", "mass", "num_particles", "type"],
    # "particle_normalisation": particle_normalizer,
    "split_fraction": [TRAIN_SPLIT, 1 - TRAIN_SPLIT, 0],
    "download": True
}

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    os.system("pip install torch-geometric torch-cluster")

    # JetNet data download args
    parser = argparse.ArgumentParser(description="Train LorentzFMNet on JetNet dataset")
    parser.add_argument("--out_dir", default="/mnt/data/output")

    parser.add_argument("--jet_types", type=str, nargs="+", default=data_args["jet_type"],
                        help="List of jet types to train on (e.g., 'g', 'q', 't')")
    parser.add_argument("--data_dir", type=str, default=data_args["data_dir"],
                        help="Directory to store the JetNet dataset")
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"],
                        help="Number of particles to consider in each jet")
    parser.add_argument("--split_fraction", type=float, nargs=3, default=data_args["split_fraction"],
                        help="Fraction of data to use for train, validation, and test splits")
    
    
    # Network hyperparameters
    parser.add_argument("--n_hidden", type=int, default=64, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--c_weight", type=float, default=1.0, help="Weight for the c parameter in the network")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--train-sample-size", type=int, default=50_000, help="Number of training samples to use")

    # Integration
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--ode_solver", type=str, choices=[m.name for m in ode_solver_methods], default="euler",
                        help="ODE solver method to use")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")
    
    args = parser.parse_args()

    # Make folders if they do not exist
    make_clear_folder(f"{args.out_dir}/figs")
    make_clear_folder(f"{args.out_dir}/logs")

    X_train = JetNet(
        jet_type=args.jet_types,
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        particle_features=JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1],
        jet_features=data_args["jet_features"],
        split_fraction=args.split_fraction,
        split="train",
    )
    X_test = JetNet(
        jet_type=args.jet_types,
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        particle_features=JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1],
        jet_features=data_args["jet_features"],
        split_fraction=args.split_fraction,
        split="valid",
    )
    print(f"{len(X_train)=}, {len(X_test)=}")
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train)
    print(f"{X_train_particle_transformed.shape=}")

    # Normalize the features
    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())
    e_c_mirrored = np.concatenate([e_c, -e_c])
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())
    final_scale = min([anderson(data).fit_result.params.scale for data in [e_c_mirrored, p_x, p_y, p_z]])

    model = LorentzFMNet(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        c_weight=args.c_weight
    ).to(device)

    X_train_particle_transformed = (1/final_scale) * X_train_particle_transformed

    X_train_loaded = DataLoader(
        X_train_particle_transformed,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)
    
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.num_epochs):
        epoch_loss = []

        for i, data in enumerate(X_train_loaded):
            x_0 = torch.randn_like(data).to(device)
            x_1 = data.to(device)

            t = torch.rand(x_0.shape[0], device=device).view(-1, 1, 1)  # Reshape t to match the expected input shape
            x_t = (1 - t) * x_0 + t * x_1  # Linear interpolation
            dx_t = x_1 - x_0
            optimizer.zero_grad()

            loss = nn.MSELoss()(model(x_t, t), dx_t)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        losses.append(np.mean(epoch_loss))
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}")
    
    sns.lineplot(x=range(len(losses)), y=losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(f"{args.out_dir}/figs/training_loss.png")

    with open(f"{args.out_dir}/logs/training_loss.csv", "w") as f:
        f.write("epoch,loss\n")
        for epoch, loss in enumerate(losses):
            f.write(f"{epoch},{loss}\n")
    # Save the model
    torch.save(model.state_dict(), f"{args.out_dir}/model.pth")

    samples = []
    with torch.no_grad():
        model.eval()
        times = torch.linspace(0, 1, args.integration_steps + 1).to(device)
        x = torch.randn(args.n_samples, NUM_PARTICLES, 4).to(device)

        for start_idx in range(0, args.n_samples, args.batch_size):
            end_idx = min(start_idx + args.batch_size, args.n_samples)
            x_batch = x[start_idx:end_idx]

            for i in range(args.integration_steps):
                x_batch = model.step(x_batch, times[i], times[i + 1], method=ode_solver_methods[args.ode_solver])
            samples.append(x_batch)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    samples = torch.cat(samples, dim=0)
    
    # Save up to 1000 random generated samples
    rand_idx = random.randint(0, args.n_samples - 1000)
    torch.save(samples[rand_idx:rand_idx+1000], f"{args.out_dir}/samples_cartesian_1000.pt")

    polar_gen_features = cartesian_to_EtaPhiPtE(x).to(device)
    x_test = (X_test[:args.n_samples][0]).to(device)

    # Metrics
    eval_info = {}
    eval_info["cov_mmd"] = eval.cov_mmd(
        real_jets=x_test[:, :, :3].to(device),
        gen_jets=polar_gen_features
    )
    eval_info["fpd"] = eval.fpd(
        real_features=x_test.reshape((-1, 4)).to(device),
        gen_features=polar_gen_features.reshape((-1, 4)).to(device),
        seed=RANDOM_SEED
    )

    # TODO
    eval_info["fpnd_g"] = eval.fpnd(
        jets=polar_gen_features[:, :, :3],
        jet_type="g",
        use_tqdm=False
    )

    jets1 = polar_gen_features[:, :, :]
    jets2 = X_test[:][0]
    eval_info["w1efp"] = eval.w1efp(
        jets1=jets1,
        jets2=jets2,
        use_particle_masses=True
    )
    eval_info["w1m"] = eval.w1m(
        jets1=jets1,
        jets2=jets2,
    )
    eval_info["w1p"] = eval.w1p(
        jets1=jets1,
        jets2=jets2,
    )
    with open(f"{args.out_dir}/eval_info.pkl", "wb") as f:
        pickle.dump(eval_info, f)
    print("Evaluation metrics saved to eval_info.pkl")