import pickle
import datetime
import argparse
import os
import shutil

import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import anderson

from jetnet.datasets import JetNet

from models.ConditionalLEFlowMatching import JetFMGenerator
from util import jet_attributes
from generate_samples import generate_samples
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.file_management import make_clear_folder

RANDOM_SEED = 42

MASK = True
NUM_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7


data_args = {
    "jet_type": ["g", "q", "t"],
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
    parser.add_argument("--n_train_samples", type=int, default=1000_000, help="Number of training samples to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")

    # Integration
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")
    
    args = parser.parse_args()

    # Make folders if they do not exist
    out_dir = f"{args.out_dir}/{str(datetime.datetime.now()).replace(' ', '_')}-{''.join(args.jet_types)}jets-{args.num_epochs}epochs-{args.n_layers}layers-{args.integration_steps}steps"
    make_clear_folder(out_dir)
    make_clear_folder(f"{out_dir}/models")
    make_clear_folder(f"{out_dir}/logs")
    make_clear_folder(f"{out_dir}/gen")
    print(f"Output directory: {out_dir}")

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
    with open(f"{out_dir}/gen/x_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    print(f"{len(X_train)=}, {len(X_test)=}")
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train)

    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())
    scales = [np.std(e_c), np.std(p_x), np.std(p_y), np.std(p_z)]
    final_scale = np.mean(scales)
    
    with open(f"{out_dir}/logs/final_scale.txt", "w") as f:
        f.write(f"{final_scale}\n")
    X_train_particle_transformed[:, :, :4] = (1/final_scale) * X_train_particle_transformed[:, :, :4]

    model: JetFMGenerator = JetFMGenerator(
        n_particles=args.num_particles,
        n_layers=args.n_layers,
        c=args.c_weight
    ).to(device)
    torch.save(model.state_dict(), f"{out_dir}/models/model_initial.pth")

    jet_info = X_train[:][1].to(device)
    encoded_jet_types = jet_attributes.one_hot_enc_jet_type(jet_info[:, 4].long()).to(device)
    jet_info_cropped = jet_info[:, :4]  # Keep only the first 4 features (eta, p_t, mass, num_particles)
    jet_info = torch.cat([encoded_jet_types, jet_info_cropped], dim=-1).to(device)

    X_train_loaded = DataLoader(
        list(zip(X_train_particle_transformed, jet_info))[:args.n_train_samples],
        batch_size=args.batch_size,
        shuffle=True
    )
    
    losses = []

    lr = 1e-3
    weight_decay = 1e-2
    warmup_pct = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = args.num_epochs * len(X_train_loaded)
    warmup_steps = int(warmup_pct * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing after warm-up
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    current_step = 0
    for epoch in range(args.num_epochs):
        epoch_loss = []

        for i, data in enumerate(X_train_loaded):
            optimizer.zero_grad()

            jet_info = data[1].to(device)
            x_1 = data[0].to(device)[:, :, :4]
            true_masks = data[0].to(device)[:, :, 4] if MASK else None
            x_0 = torch.randn_like(x_1, device=device)  # Sample random initial state

            t = torch.rand(x_0.shape[0], device=device)
            t_viewed = t.view(-1, 1, 1)
            x_t = (1 - t_viewed) * x_0 + t_viewed * x_1  # Linear interpolation
            dx_t = x_1 - x_0

            pred = model.forward(x_t, t, jet_info, true_masks)
            loss = nn.MSELoss()(pred, dx_t)
            loss.backward()
            model.clip_gradients()
            optimizer.step()

            scheduler.step()
            current_step += 1

            epoch_loss.append(loss.item())
        
            if i % (args.batch_size * 10) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(X_train_loaded)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                print(f"dx_t: mean={dx_t.abs().mean()}, std={dx_t.abs().std()}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2       # Tensors currently live
                    reserved = torch.cuda.memory_reserved() / 1024**2         # Memory reserved by PyTorch's caching allocator
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # Peak allocation during program
                    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Peak: {max_allocated:.2f} MB")

        losses.append(np.mean(epoch_loss))
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}, LR: {current_lr:.6f}")
    
    with open(f"{out_dir}/logs/training_loss.csv", "w") as f:
        f.write("epoch,loss\n")
        for epoch, loss in enumerate(losses):
            f.write(f"{epoch},{loss}\n")

    torch.save(model.state_dict(), f"{out_dir}/models/final_model.pth")
    with open(f"{out_dir}/logs/args.txt", "w") as f:
        f.write(f"CLI args:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.close()
    
    generate_samples(
        model=model,
        device=device,
        out_dir=out_dir,
        num_particles=args.num_particles,
        num_particle_features=NUM_PARTICLE_FEATURES,
        final_scale=final_scale,
        integration_steps=args.integration_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        jet_types=len(args.jet_types)
    )

    # move everything in out/ to final_out/
    final_out_dir = f"{args.out_dir}/final_out"
    make_clear_folder(final_out_dir)
    for item in os.listdir(out_dir):
        src_path = os.path.join(out_dir, item)
        dst_path = os.path.join(final_out_dir, item)
        if os.path.isdir(src_path):
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    print(f"Training complete. Output saved to {final_out_dir}")