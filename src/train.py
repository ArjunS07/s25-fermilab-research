import pickle
import argparse
import logging

import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from jetnet.utils import cartesian_to_EtaPhiPtE

from models.NewLEFM import LEJetGeneratorFM
from models.FlowMatchingMLP import FlowMatchingMLP
from models.Week7EGNN import JetFlowMatcher
from util import jet_attributes
from util.jet_attributes import NUM_CLASSES
from jet_attr_model import get_model_pth_path
from util.distributions import gen_initial_distribution
from util.coordinates import transform_rel_particle_coordinates_to_cartesian, jacobian_epp_etaphipte
from util.file_management import make_clear_folder
from util.viz import generate_model_vector_field
from data import data_args, get_data_path

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
    parser.add_argument("--output_path", type=str, default="/mnt/data/output", help="Path to save the output files")

    # Data
    parser.add_argument("--jet_types", type=str, nargs="+", default=data_args["jet_type"])
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"], help="Number of particles in each jet")
    parser.add_argument("--mask", type=bool, default=True, help="Use mask for particles")

    # Network hyperparameters
    parser.add_argument("--model_type", type=str, default="Week7EGNN", choices=["LEJetGeneratorFM", "FlowMatchingMLP", "Week7EGNN"], help="Type of model to use")
    parser.add_argument("--use_residual_update", type=bool, default=False, help="Use residual update in model forward pass")
    parser.add_argument("--n_hidden", type=int, default=64, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the network")
    
    # Training
    parser.add_argument("--use_distributed", type=bool, default=False, help="Use distributed for training")
    parser.add_argument("--multi_core", type=bool, default=False, help="Use multiple cores for training")
    parser.add_argument("--accumulate_gradients", type=bool, default=False, help="Accumulate gradients over multiple batches" )
    parser.add_argument("--n_train_samples", type=int, default=1000_000, help="Number of training samples to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--epoch_frac", type=float, default=1.0, help="Fraction of training dataset to use per epoch")

    parser.add_argument("--x_1_translation", type=float, default=0.0, help="Translation to apply to x_1 during training")

    # Integration
    parser.add_argument("--n_samples", type=int, default=5_000, help="Number of samples to generate during inference")
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
    if args.use_distributed:
        accelerator.wait_for_everyone()

    data_path = get_data_path(args.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(f"{data_path}/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)


    model_output_path = f"{args.output_path}/train"
    make_clear_folder(model_output_path)

    if not args.use_distributed or accelerator.is_main_process:
        with open(f"{model_output_path}/args.txt", "w") as f:
            f.write(f"CLI args:\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.close()

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
        with open(f"{model_output_path}/scale.txt", "w") as f:
            f.write(f"{final_scale}\n")
    X_train_particle_transformed[:, :, :4] = (1/final_scale) * X_train_particle_transformed[:, :, :4]
    
    if args.model_type == "Week7EGNN":
        model: JetFlowMatcher = JetFlowMatcher(
            max_num_jet_types=NUM_CLASSES,
            max_particles=args.num_particles,
            num_layers=args.n_layers,
            hidden_dim=args.n_hidden,
            use_residual_update=args.use_residual_update,
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
    
    make_clear_folder(f"{model_output_path}/models")
    if args.use_distributed:
        if accelerator.is_main_process:
            accelerator.save_model(model, f"{model_output_path}/models/initial_model.pth")
    else:
        torch.save(model.state_dict(), f"{model_output_path}/models/initial_model.pth")
    train_jet_info = X_train[:][1].to(device)
    encoded_jet_types = jet_attributes.one_hot_enc_jet_type(train_jet_info[:, 4].long()).to(device)
    if args.model_type == "Week7EGNN":
        train_jet_n_particles = train_jet_info[:, 3]
        train_jet_pt_mass = train_jet_info[:, 1:3]
        # Model will leave pt and mass as scalars
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

    # Annealed cosine learning rate with warmup
    lr = 1e-3
    weight_decay = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_fraction = args.epoch_frac
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))
    
    # warmup_pct = 0.1
    # steps_per_epoch = (samples_per_epoch + args.batch_size - 1) // args.batch_size  # Ceiling division
    # total_steps = args.num_epochs * steps_per_epoch
    # warmup_steps = int(warmup_pct * total_steps)
    # def lr_lambda(current_step):
    #     if current_step < warmup_steps:
    #         # Linear warm-up
    #         return float(current_step) / float(max(1, warmup_steps))
    #     else:
    #         # Cosine annealing after warm-up
    #         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    #         lr = 0.5 * (1.0 + np.cos(np.pi * progress))
    #         return max(lr, 1e-6) # Ensure nonzero
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # current_step = 0

    # For OT objective
    sigma_min = 1e-4

    if args.use_distributed:
        model, optimizer = accelerator.prepare(model, optimizer)
    
    # print(f"Starting training for {args.num_epochs} epochs with {steps_per_epoch} steps per epoch")
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
            x_1 += args.x_1_translation
            true_masks = batch_particle_info[:, :, 4] if args.mask else None
            x_0 = gen_initial_distribution(x_1=x_1)
            x_0 = x_0.to(device)
            
            if true_masks is not None:
                # multiply x_1 for redundancy, training data should have it masked by default
                x_1 = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES) * x_1
                # important - x0 is a noisy normal distribution by default
                x_0 = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES) * x_0

            # Logit-normal samplign of t to focus around t=0.5 which is hardest
            # https://github.com/UNITES-Lab/FlowTS
            # t = torch.sigmoid(torch.randn(x_0.shape[0]))
            t = torch.rand(x_0.shape[0])
            if not args.use_distributed:
                t = t.to(device)
            t_viewed = t.view(-1, 1, 1)

            # The model takes in Cartesian coordinates, and x_t is in Cartesian coordinates
            x_t = (1 - (1-sigma_min)*t_viewed)*x_0 + t_viewed * x_1
            Jacobian_x_t = jacobian_epp_etaphipte(x_t)

            conditional_u_t_cartesian = x_1 - ((1-sigma_min)*x_0)
            conditional_u_t_polar = torch.einsum('...ij, ...j->...i', Jacobian_x_t, conditional_u_t_cartesian)

            pred_cartesian = model.forward(x=x_t, t=t, jet_conditions=batch_jet_info, mask=true_masks)

            # result[..., i] = \sum_j Jacobian[..., i, j] * vector[..., j]
            pred_polar = torch.einsum('...ij, ...j->...i', Jacobian_x_t, pred_cartesian)

            if true_masks is not None:
                conditional_u_t_polar = conditional_u_t_polar * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
                pred_polar = pred_polar * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)

            cartesian_loss = (conditional_u_t_cartesian - pred_cartesian).square()
            polar_loss = (conditional_u_t_polar - pred_polar).square()
            if true_masks is not None:
                cartesian_loss = cartesian_loss * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
                polar_loss = polar_loss * true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
                loss = (0.5 * cartesian_loss) + (0.5 * polar_loss)
                loss = loss.sum() / (true_masks.sum() * NUM_PARTICLE_FEATURES)
            else:
                loss = loss.mean()

            if args.use_distributed:
                accelerator.backward(loss)  # Use accelerator to handle backward pass
            else:
                loss.backward()

            if args.use_distributed:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                            
            optimizer.step()
            # scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            # current_step += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if i % 50 == 0 and (not args.use_distributed or accelerator.is_main_process):            # if True:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_lr = optimizer.param_groups[0]['lr']
                current_avg_loss = epoch_loss / num_batches
                print(f"{x_t.mean()=} {x_t.std()=} {x_t.min()=} {x_t.max()=}")
                print(f"{x_1.mean()=} {x_1.std()=} {x_1.min()=} {x_1.max()=}")
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

                with torch.no_grad():
                    zero_input = torch.randn_like(x_t) * 0.01
                    pred_at_zero = model.forward(x=zero_input, t=t, jet_conditions=batch_jet_info, mask=true_masks)
                    print(f"Model prediction at origin: mean={pred_at_zero.mean():.6f}, std={pred_at_zero.std():.6f}")
            
            del x_1, x_0, t, t_viewed, x_t, conditional_u_t_cartesian, conditional_u_t_polar, pred_cartesian, pred_polar, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        losses.append(epoch_loss / num_batches)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}, LR: {current_lr:.6f}")

    if not args.use_distributed or accelerator.is_main_process:
        with open(f"{model_output_path}/training_loss.csv", "w") as f:
            f.write("epoch,loss\n")
            for epoch, loss in enumerate(losses):
                f.write(f"{epoch},{loss}\n")

    if args.use_distributed:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_model(model, f"{model_output_path}/models/final_model.pth")
    else:
        torch.save(model.state_dict(), f"{model_output_path}/models/final_model.pth")
    
    if not args.use_distributed or accelerator.is_main_process:
        generate_model_vector_field(
            out_dir=model_output_path,
            final_model=model,
            jet_attr_model=jet_attributes.load_model(model_path=get_model_pth_path(args.output_path)).to(device),
            X_test=X_test,
            scale=final_scale,
            n_jet_types=len(args.jet_types),
            n_particles_per_jet=args.num_particles,
            n_features_per_particle=NUM_PARTICLE_FEATURES,
            n_viz_samples=(args.n_samples)
        )