import pickle
import datetime
import argparse
import os
import shutil
import logging
import sys

import numpy as np
import torch
import random
from torch import nn


from torch.utils.data import DataLoader

from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear

from models.NewLEFM import JetFMGenerator
from util import jet_attributes
from generate_samples import generate_samples
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.file_management import make_clear_folder
from util.memory_management import log_memory_usage
from data import data_args

RANDOM_SEED = 42

MASK = True
NUM_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # JetNet data download args
    parser = argparse.ArgumentParser(description="Train LorentzFMNet on JetNet dataset")
    parser.add_argument("--out_dir", default="/mnt/data/output")

    # Network hyperparameters
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training
    parser.add_argument("--n_train_samples", type=int, default=1000_000, help="Number of training samples to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")

    # Integration
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")
    
    args = parser.parse_args()


    # Make folders if they do not exist
    out_dir = f"{args.out_dir}/{str(datetime.datetime.now()).replace(' ', '_')}-{args.num_epochs}epochs-{args.n_layers}layers-{args.integration_steps}steps"
    make_clear_folder(out_dir)
    make_clear_folder(f"{out_dir}/models")
    make_clear_folder(f"{out_dir}/logs")
    make_clear_folder(f"{out_dir}/gen")
    print(f"Output directory: {out_dir}")

    with open("data/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("data/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train).to('cpu')

    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())
    scales = [np.std(e_c), np.std(p_x), np.std(p_y), np.std(p_z)]
    final_scale = np.mean(scales)
    
    with open(f"{out_dir}/logs/final_scale.txt", "w") as f:
        f.write(f"{final_scale}\n")
    X_train_particle_transformed[:, :, :4] = (1/final_scale) * X_train_particle_transformed[:, :, :4]
    print(f"{final_scale=}")

    model: JetFMGenerator = JetFMGenerator(
        n_particles=data_args["num_particles"],
        n_layers=args.n_layers,
    ).to(device)
    torch.save(model.state_dict(), f"{out_dir}/models/model_initial.pth")

    jet_info = X_train[:][1].to(device)
    encoded_jet_types = jet_attributes.one_hot_enc_jet_type(jet_info[:, 4].long()).to(device)
    jet_info_cropped = jet_info[:, :4]  # Keep only the first 4 features (eta, p_t, mass, num_particles)
    jet_info = torch.cat([encoded_jet_types, jet_info_cropped], dim=-1).to(device)

    losses = []

    lr = 1e-3
    weight_decay = 1e-2
    warmup_pct = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_fraction = 0.001  # Use 1% of data per epoch
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))
    steps_per_epoch = (samples_per_epoch + args.batch_size - 1) // args.batch_size  # Ceiling division

    total_steps = args.num_epochs * steps_per_epoch
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

    print(f"Starting training for {args.num_epochs} epochs with {steps_per_epoch} steps per epoch")
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0

        # Generate random indices for this epoch
        epoch_indices = torch.randperm(len(X_train_particle_transformed))[:samples_per_epoch]
        X_train_epoch = torch.utils.data.Subset(X_train_particle_transformed, epoch_indices)
        jet_info_epoch = jet_info[epoch_indices].to(device)  # Move to device once
        X_train_loaded = DataLoader(
            X_train_epoch,
            batch_size=args.batch_size,
            shuffle=False, 
            # num_workers=2 if device.type == 'cuda' else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        for i, data in enumerate(X_train_loaded):
            optimizer.zero_grad()

            actual_batch_size = data.shape[0]
            batch_jet_info = jet_info_epoch[i * args.batch_size:i * args.batch_size + actual_batch_size]
            
            x_1 = data.to(device)[:, :, :4]
            true_masks = data.to(device)[:, :, 4] if MASK else None
            x_0 = torch.randn_like(x_1, device=device)  # Sample random initial state
            if true_masks is not None:
                x_0 = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES) * x_0

            t = torch.rand(x_0.shape[0], device=device)
            t_viewed = t.view(-1, 1, 1)
            x_t = (1 - t_viewed) * x_0 + t_viewed * x_1  # Linear interpolation
            dx_t = x_1 - x_0

            pred = model.forward(x_t, t, batch_jet_info, true_masks)
            # breakpoint()
            loss = nn.MSELoss()(pred, dx_t)
        
            loss.backward()
            model.clip_gradients()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            current_step += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i % 50 == 0:
            # if True:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_lr = optimizer.param_groups[0]['lr']
                current_avg_loss = epoch_loss / num_batches
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(X_train_loaded)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                print(f"dx_t: mean={dx_t.abs().mean()}, std={dx_t.abs().std()}")
                # log_memory_usage()
            
            del x_1, x_0, t, t_viewed, x_t, dx_t, pred, loss

        losses.append(epoch_loss / num_batches)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}, LR: {current_lr:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get validation metrics
        if epoch % 50 == 0:
            print(f"Generating samples for epoch {epoch+1}...")
            model.eval()
            with torch.no_grad():
                make_clear_folder(f"{out_dir}/gen/epoch_{epoch+1}")
                try:
                    val_samples = generate_samples(
                        model=model,
                        device=device,
                        out_dir=f"{out_dir}/gen/epoch_{epoch+1}",
                        num_particles=data_args["num_particles"],
                        num_particle_features=NUM_PARTICLE_FEATURES,
                        final_scale=final_scale,
                        integration_steps=16,
                        n_samples=max(args.n_samples // 100, 50),
                        batch_size=args.batch_size,
                        n_jet_types=len(data_args["jet_type"])
                    )
                    torch.cuda.synchronize()
                    del val_samples
                except Exception as e:
                    print(f"Error generating samples for epoch {epoch+1}: {e}")
            model.train()
    
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
    
    make_clear_folder(f"{out_dir}/gen/samples")
    generate_samples(
        model=model,
        device=device,
        out_dir=f"{out_dir}/gen/samples",
        num_particles=data_args["num_particles"],
        num_particle_features=NUM_PARTICLE_FEATURES,
        final_scale=final_scale,
        integration_steps=args.integration_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_jet_types=len(data_args["jet_type"])
    )

    # move everything in out/ to final_out/
    final_out_dir = f"{args.out_dir}/final_out"
    make_clear_folder(final_out_dir)
    for item in os.listdir(out_dir):
        src_path = os.path.join(out_dir, item)
        dst_path = os.path.join(final_out_dir, item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
    print(f"Training complete. Output saved to {final_out_dir}") 