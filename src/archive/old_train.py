import datetime
import os
import argparse
import pickle

import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import anderson

from jetnet.datasets import JetNet
import jetnet.evaluation as eval
from jetnet.utils import cartesian_to_EtaPhiPtE

from models.FMLorentzNet import LorentzFMNet, ode_solver_methods
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.file_management import make_clear_folder
from util.distributions import sample_massless_4momentum_clouds

RANDOM_SEED = 42

MASK = False
NUM_PARTICLES = 30
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7


data_args = {
    "jet_type": ["g"],
    "data_dir": "datasets/jetnet",
    "num_particles": NUM_PARTICLES,
    "particle_features": (
        JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1]
    ),
    "jet_features": ["eta", "pt", "mass", "num_particles", "type"],
    "split_fraction": [TRAIN_SPLIT, 1 - TRAIN_SPLIT, 0],
    "download": True
}

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--c_weight", type=float, default=1.0, help="Weight for the c parameter in the network")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")

    # Integration
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--ode_solver", type=str, choices=[m.name for m in ode_solver_methods], default="euler",
                        help="ODE solver method to use")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")
    
    args = parser.parse_args()

    # Make folders if they do not exist
    out_dir = f"{args.out_dir}/{str(datetime.datetime.now()).replace(' ', '_')}"
    make_clear_folder(out_dir)
    make_clear_folder(f"{out_dir}/models")
    make_clear_folder(f"{out_dir}/logs")
    make_clear_folder(f"{out_dir}/gen")

    X_train = JetNet(
        jet_type=args.jet_types,
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        particle_features=JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1],
        jet_features=data_args["jet_features"],
        split_fraction=args.split_fraction,
        split="train",
        download=True
    )
    X_test = JetNet(
        jet_type=args.jet_types,
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        particle_features=JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1],
        jet_features=data_args["jet_features"],
        split_fraction=args.split_fraction,
        split="valid",
        download=True
    )
    print(f"{len(X_train)=}, {len(X_test)=}")
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train)

    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())
    e_c_mirrored = np.concatenate([e_c, -e_c])
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())
    final_scale = min([anderson(data).fit_result.params.scale for data in [e_c_mirrored, p_x, p_y, p_z]])
    with open(f"{out_dir}/logs/final_scale.txt", "w") as f:
        f.write(f"{final_scale}\n")
        f.close()
    X_train_particle_transformed = (1/final_scale) * X_train_particle_transformed

    model = LorentzFMNet(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        c_weight=args.c_weight,
        device=device
    ).to(device)
    torch.save(model.state_dict(), f"{out_dir}/models/model_initial.pth")


    X_train_loaded = DataLoader(
        X_train_particle_transformed,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)
    
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Beginning training")

    for epoch in range(args.num_epochs):
        epoch_loss = []

        for i, data in enumerate(X_train_loaded):
            # print(data.shape)
            x_0 = sample_massless_4momentum_clouds(n_clouds=len(data), cloud_size=NUM_PARTICLES, device=device).to(device)
            # print(x_0.shape)
            x_1 = data.to(device)[:, :, :4]

            t = torch.rand(x_0.shape[0], device=device).view(-1, 1, 1) 
            x_t = (1 - t) * x_0 + t * x_1  # Linear interpolation
            dx_t = x_1 - x_0
            optimizer.zero_grad()

            loss = nn.MSELoss()(model.forward(x_t, t), dx_t)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        
            if i % 500 == 0:
                print(f"dx_t: mean={dx_t.abs().mean()}, std={dx_t.abs().std()}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2       # Tensors currently live
                    reserved = torch.cuda.memory_reserved() / 1024**2         # Memory reserved by PyTorch's caching allocator
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # Peak allocation during program
                    print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}. Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Peak: {max_allocated:.2f} MB")


        losses.append(np.mean(epoch_loss))
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}")
    
    with open(f"{out_dir}/logs/training_loss.csv", "w") as f:
        f.write("epoch,loss\n")
        for epoch, loss in enumerate(losses):
            f.write(f"{epoch},{loss}\n")

    torch.save(model.state_dict(), f"{out_dir}/models/final_model.pth")
    print("Model saved to final_model.pth")
    with open(f"{out_dir}/logs/model_info.txt", "w") as f:
        f.write(f"Model hyperparameters:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"Final scale: {final_scale}\n")
        f.close()

    samples = []
    with torch.no_grad():
        model.eval()
        times = torch.linspace(0, 1, args.integration_steps + 1).to(device)
        x = sample_massless_4momentum_clouds(n_clouds=len(data), cloud_size=NUM_PARTICLES, device=device).to(device)

        for start_idx in range(0, args.n_samples, args.batch_size):
            end_idx = min(start_idx + args.batch_size, args.n_samples)
            x_batch = x[start_idx:end_idx]

            for i in range(args.integration_steps):
                x_batch = model.step(x_batch, times[i], times[i + 1], method=ode_solver_methods[args.ode_solver])
            samples.append(x_batch)
            if start_idx % 100 == 0:
                print(f"Processed samples {start_idx} to {end_idx}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done generating samples! Saving samples")
    samples = final_scale * torch.cat(samples, dim=0)
    torch.save(samples, f"{out_dir}/gen/samples_cartesian.pt")
    
    try:
        polar_gen_features = cartesian_to_EtaPhiPtE(samples).to(device)
        x_test = (X_test[:args.n_samples][0]).to(device)
    except Exception as e:
        breakpoint()

    # Metrics
    eval_info = {}
    eval_info["cov_mmd"] = eval.cov_mmd(
        real_jets=x_test[:, :, :3].to(device),
        gen_jets=polar_gen_features
    )
    eval_info["fpd"] = eval.fpd(
        real_features=x_test.reshape((-1, NUM_PARTICLE_FEATURES)).to(device),
        gen_features=polar_gen_features.reshape((-1, NUM_PARTICLE_FEATURES)).to(device),
        seed=RANDOM_SEED
    )

    # Requires torch geometric
    # eval_info["fpnd_g"] = eval.fpnd(
    #     jets=polar_gen_features[:, :, :3],
    #     jet_type="g",
    #     use_tqdm=False
    # )

    # Don't include mass
    jets1 = polar_gen_features[:, :, :3]
    jets2 = X_test[:][0]
    try:
        eval_info["w1efp"] = eval.w1efp(
            jets1=jets1,
            jets2=jets2,
        )
    except Exception as e:
        print(f"Error calculating W1 EFP metrics: {e}")
    try:
        eval_info["w1m"] = eval.w1m(
            jets1=jets1,
            jets2=jets2,
        )
        eval_info["w1p"] = eval.w1p(
            jets1=jets1,
            jets2=jets2,
        )
    except Exception as e:
        print(f"Error calculating W1 metrics: {e}")

    with open(f"{out_dir}/logs/eval_info.pkl", "wb") as f:
        pickle.dump(eval_info, f)
    print("Evaluation metrics saved to eval_info.pkl")