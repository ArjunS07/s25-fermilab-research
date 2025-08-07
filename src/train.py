import pickle
import datetime
import argparse
import os
import shutil
import logging

import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from models.NewLEFM import LEJetGeneratorFM
from models.FlowMatchingMLP import FlowMatchingMLP
from models.Week7EGNN import JetFlowMatcher
from util.jet_attributes import NUM_CLASSES

from generate_samples import generate_samples
from util import jet_attributes
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.file_management import make_clear_folder
from data import data_args

RANDOM_SEED = 42
MAX_N_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, jet_info, particle_data):
        self.jet_info = jet_info
        self.particle_data = particle_data
    
    def __len__(self):
        return len(self.particle_data)
    
    def __getitem__(self, idx):
        return self.jet_info[idx], self.particle_data[idx]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # JetNet data download args
    parser = argparse.ArgumentParser(description="Train LEJetGeneratorFM on JetNet dataset")
    parser.add_argument("--out_dir", default="/mnt/data/output")
    parser.add_argument("--process_id", type=str, default="abcd", help="Process ID for distributed training")

    # Data
    parser.add_argument("--jet_types", type=str, nargs="+", default=data_args["jet_type"])
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"], help="Number of particles in each jet")
    parser.add_argument("--mask", type=bool, default=True, help="Use mask for particles")

    # Network hyperparameters
    parser.add_argument("--model_type", type=str, default="Week7EGNN", choices=["LEJetGeneratorFM", "FlowMatchingMLP", "Week7EGNN"], help="Type of model to use")
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the network")
    
    # Training
    parser.add_argument("--use_distributed", type=bool, default=False, help="Use distributed for training")
    parser.add_argument("--multi_core", type=bool, default=False, help="Use multiple cores for training")
    parser.add_argument("--accumulate_gradients", type=bool, default=False, help="Accumulate gradients over multiple batches" )
    parser.add_argument("--n_train_samples", type=int, default=1000_000, help="Number of training samples to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--epoch_frac", type=float, default=0.2, help="Fraction of training dataset to use per epoch")

    parser.add_argument("--x_1_translation", type=float, default=0.0, help="Translation to apply to x_1 during training")

    # Integration
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")
    
    args = parser.parse_args()

    if args.use_distributed:
        accelerator = Accelerator()
        device = accelerator.device
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Make folders if they do not exist
    out_dir = f"{args.out_dir}/{args.process_id}-{args.model_type}-{args.num_epochs}epochs-{args.batch_size}batch-{args.n_layers}layers-{args.integration_steps}steps"
    if not args.use_distributed or (accelerator is not None and accelerator.is_main_process):
        make_clear_folder(out_dir)
        make_clear_folder(f"{out_dir}/models")
        make_clear_folder(f"{out_dir}/logs")
        make_clear_folder(f"{out_dir}/gen")
        print(f"Output directory: {out_dir}")
    if args.use_distributed:
        accelerator.wait_for_everyone()
    # Load data
    with open("data/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("data/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train).to('cpu')
    X_train_particle_transformed = X_train_particle_transformed[:args.n_train_samples]
    if args.num_particles < MAX_N_PARTICLES:
        # Particles are, by default, ordered by p_t. take the n highest pt particles in each jet
        X_train_particle_transformed = X_train_particle_transformed[:, :args.num_particles, :]
    
    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())
    scales = [np.std(e_c), np.std(p_x), np.std(p_y), np.std(p_z)]
    final_scale = np.mean(scales)
    if not args.use_distributed or (accelerator is not None and accelerator.is_main_process):
        with open(f"{out_dir}/logs/final_scale.txt", "w") as f:
            f.write(f"{final_scale}\n")
    X_train_particle_transformed[:, :, :4] = (1/final_scale) * X_train_particle_transformed[:, :, :4]
    
    if args.model_type == "Week7EGNN":
        model: JetFlowMatcher = JetFlowMatcher(
            max_num_jet_types=NUM_CLASSES,
            max_particles=args.num_particles,
            num_layers=args.n_layers,
            hidden_dim=args.n_hidden,
        ).to(device)
    elif args.model_type == "LEJetGeneratorFM":
        model: LEJetGeneratorFM = LEJetGeneratorFM(
            n_layers=args.n_layers,
            n_particles=args.num_particles
        ).to(device)
    elif args.model_type == "FlowMatchingMLP":
        model = FlowMatchingMLP(
            n_particles=args.num_particles,
            particle_dim=NUM_PARTICLE_FEATURES,
            global_dim=64,
            n_layers=args.n_layers,
            hidden_dim=args.n_hidden,
            n_jet_types=len(data_args["jet_type"]),
            time_embed_dim=64,
        ).to(device)
    
    if args.use_distributed:
        if accelerator.is_main_process:
            accelerator.save_model(model, f"{out_dir}/models/model_initial.pth")
    else:
        torch.save(model.state_dict(), f"{out_dir}/models/model_initial.pth")
    train_jet_info = X_train[:][1].to(device)
    encoded_jet_types = jet_attributes.one_hot_enc_jet_type(train_jet_info[:, 4].long()).to(device)
    if args.model_type == "Week7EGNN":
        train_jet_n_particles = train_jet_info[:, 3]
        train_jet_info = torch.cat([
            encoded_jet_types, train_jet_n_particles.unsqueeze(-1)
        ], dim=-1).to(device)
    else:
        jet_info_cropped = train_jet_info[:, :4]  # Keep only the first 4 features (eta, p_t, mass, num_particles)
        train_jet_info = torch.cat([encoded_jet_types, jet_info_cropped], dim=-1).to(device)
    if args.num_particles < MAX_N_PARTICLES:
        # clamp the number of particles to args.num_particles
        train_jet_info[:, 3] = train_jet_info[:, 3].clamp(max=args.num_particles)
    train_jet_info = train_jet_info[:args.n_train_samples]
    losses = []

    lr = 5e-3
    weight_decay = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_fraction = args.epoch_frac
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))

    warmup_pct = 0.1
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
    print(f"Current date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.use_distributed:
        model, optimizer = accelerator.prepare(model, optimizer)
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0

        # Generate random indices for this epoch
        epoch_indices = torch.randperm(len(X_train_particle_transformed))[:samples_per_epoch]
        X_train_epoch = torch.utils.data.Subset(X_train_particle_transformed, epoch_indices)
        train_jet_info_epoch = train_jet_info[epoch_indices]

        print(torch.sum(train_jet_info_epoch, dim=0))

        paired_dataset = PairedDataset(train_jet_info_epoch, X_train_epoch)
        train_loader = DataLoader(
            paired_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False
        )
        if args.use_distributed:
            train_loader = accelerator.prepare_data_loader(train_loader)

        for i, (batch_jet_info, batch_particle_info) in enumerate(train_loader):
            optimizer.zero_grad()

            if not args.use_distributed:
                batch_jet_info = batch_jet_info.to(device)
                batch_particle_info = batch_particle_info.to(device)

            x_1 = batch_particle_info[:, :, :4]
            # translate every value in x_1 by +10 
            x_1 += args.x_1_translation
            true_masks = batch_particle_info[:, :, 4] if args.mask else None
            x_0 = torch.randn_like(x_1, device=device)
            x_0 = 0.5 + x_0
            
            if true_masks is not None:
                # multiply x_1 for redundancy, training data should have it masked by default
                x_1 = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES) * x_1

                # important - x0 is a noisy normal distribution by default
                x_0 = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES) * x_0

            t = torch.rand(x_0.shape[0])
            if not args.use_distributed:
                t = t.to(device)
            t_viewed = t.view(-1, 1, 1)

            # Should already be 0 in masked regions due to x0 and x1 being masked
            x_t = (1 - t_viewed) * x_0 + t_viewed * x_1  # Linear interpolation
            dx_t = x_1 - x_0
            if true_masks is not None:
                dx_t = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES) * dx_t
            pred = model.forward(x=x_t, t=t, jet_conditions=batch_jet_info, mask=true_masks)
            # only take loss over unmasked parts
            if true_masks is not None:
                pred = pred * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
                dx_t = dx_t * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
            loss = (pred - dx_t).square()
            if true_masks is not None:
                loss = loss * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
                loss = loss.sum() / true_masks.sum()
            else:
                loss = loss.mean(dim=-1)

            if args.use_distributed:
                accelerator.backward(loss)  # Use accelerator to handle backward pass
            else:
                loss.backward()
            # for name, param in model.named_parameters():
                # if param.grad is not None:
                    # print(name, param.grad.abs().mean())

            if args.use_distributed:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            current_step += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if i % 50 == 0 and (not args.use_distributed or accelerator.is_main_process):            # if True:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_lr = optimizer.param_groups[0]['lr']
                current_avg_loss = epoch_loss / num_batches
                print(f"{x_t.mean()=} {x_t.std()=} {x_t.min()=} {x_t.max()=}")
                print(f"{dx_t.mean()=} {dx_t.std()=} {dx_t.min()=} {dx_t.max()=}")
                print(f"{x_1.mean()=} {x_1.std()=} {x_1.min()=} {x_1.max()=}")
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            
            del x_1, x_0, t, t_viewed, x_t, dx_t, pred, loss

        losses.append(epoch_loss / num_batches)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}, LR: {current_lr:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get validation metrics
        if epoch % 50 == 0 and (not args.use_distributed or accelerator.is_main_process):
            print(f"Generating samples for epoch {epoch+1}...")
            model.eval()
            with torch.no_grad():
                make_clear_folder(f"{out_dir}/gen/epoch_{epoch+1}")
                try:
                    val_samples = generate_samples(
                        model=model,
                        device=device,
                        out_dir=f"{out_dir}/gen/epoch_{epoch+1}",
                        num_particles=args.num_particles,
                        num_particle_features=NUM_PARTICLE_FEATURES,
                        final_scale=final_scale,
                        integration_steps=16,
                        n_samples=max(args.n_samples // 100, 50),
                        batch_size=args.batch_size,
                        n_jet_types=len(args.jet_types)
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    del val_samples
                except Exception as e:
                    print(f"Error generating samples for epoch {epoch+1}: {e}")
            model.train()
    
    if not args.use_distributed or accelerator.is_main_process:
        with open(f"{out_dir}/logs/training_loss.csv", "w") as f:
            f.write("epoch,loss\n")
            for epoch, loss in enumerate(losses):
                f.write(f"{epoch},{loss}\n")

    if args.use_distributed:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_model(model, f"{out_dir}/models/final_model.pth")
    else:
        torch.save(model.state_dict(), f"{out_dir}/models/final_model.pth")
    
    if not args.use_distributed or accelerator.is_main_process:
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
            num_particles=args.num_particles,
            num_particle_features=NUM_PARTICLE_FEATURES,
            final_scale=final_scale,
            integration_steps=args.integration_steps,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            n_jet_types=len(args.jet_types)
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