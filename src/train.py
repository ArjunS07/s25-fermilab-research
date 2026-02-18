import pickle
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.jet_attributes import NUM_CLASSES
from jet_attr_model import get_model_pth_path
from util.distributions import gen_initial_distribution, time_dist
from util.coordinates import transform_rel_particle_coordinates_to_cartesian, jacobian_epp_etaphipte
from jetnet.utils import EtaPhiPtE_to_cartesian, cartesian_to_EtaPhiPtE
from util.file_management import make_clear_folder
from util.viz import generate_model_vector_field
from util.metrics import run_save_metrics
from util.cfg import null_vector_like
# from util.boost_equiv import boost_to_com_frame
from generate_samples import generate_samples
from data import data_args, get_data_path
from util.mask_helpers import mean_std_masked_tensor

RANDOM_SEED = 42
MAX_N_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7

# SCALE = 2000

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, jet_info, particle_data, x_0_cache=None):
        self.jet_info = jet_info
        self.particle_data = particle_data
        self.x_0_cache = x_0_cache   # optional (N, P, 4) tensor of ICP-aligned priors

    def __len__(self):
        return len(self.particle_data)

    def __getitem__(self, idx):
        x_0 = self.x_0_cache[idx] if self.x_0_cache is not None else torch.zeros(1)
        return self.jet_info[idx], self.particle_data[idx], x_0

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

    # Network hyperparameters
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers in the network")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connections in the network")
    
    # Training
    parser.add_argument("--n_train_samples", type=int, default=1000_000, help="Number of training samples to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--target_batch_size", type=int, default=256, help="Virtual batch size for accumulation of gradients for backprop")
    parser.add_argument('--cfg_null_dropout_rate', type=float, default=0.2, help='Probability of dropping out jet information and using null vector')
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--epoch_frac", type=float, default=1.0, help="Fraction of training dataset to use per epoch")
    parser.add_argument("--sigma_min", type=float, default=1e-4, help="Tolerance around target for flow map")
    parser.add_argument("--train_space", type=str, default='cartesian', choices=['cartesian', 'polar'], help="Coordinate space in which to interpolate points")
    parser.add_argument("--time_sampling", type=str, default='uniform', choices=['uniform', 'power_law', 'lognorm'], help="Method to sample time steps during training")

    # Inference
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--n_viz_samples", type=int, default=1000, help="Number of samples to generate for VF visualization")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")

    # Ablation flags (all True by default; disable with --no-<flag>)
    parser.add_argument("--use_cosine_lr", action=argparse.BooleanOptionalAction, default=True,
                        help="Use cosine annealing LR schedule (disable: --no-use_cosine_lr)")
    parser.add_argument("--use_curriculum", action=argparse.BooleanOptionalAction, default=True,
                        help="Use curriculum learning (disable: --no-use_curriculum)")
    parser.add_argument("--use_time_sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the --time_sampling method; when False, always use uniform "
                             "(disable: --no-use_time_sampling)")

    # Curriculum settings
    parser.add_argument("--curriculum_alpha_start", type=float, default=2.0,
                        help="Initial power-law exponent α (positive → oversample dense jets). "
                             "Decays linearly to 0 by the final epoch.")
    parser.add_argument("--n_curriculum_buckets", type=int, default=10,
                        help="Number of particle-count buckets for curriculum")

    # ICP cache
    parser.add_argument("--icp_cache_path", type=str, default=None,
                        help="Path to pre-computed ICP-aligned prior cache (from cache_icp.py). "
                             "When set, the cached x_0 is used during training instead of "
                             "sampling a fresh prior each step.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    data_path = get_data_path(args.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(f"{data_path}/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    model_output_path = f"{args.output_path}/train"
    make_clear_folder(model_output_path)

    with open(f"{model_output_path}/args.txt", "w") as f:
        f.write(f"CLI args:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train).to('cpu')
    X_train_particle_transformed = X_train_particle_transformed[:args.n_train_samples]
    if args.num_particles < MAX_N_PARTICLES:
        # Particles are, by default, ordered by p_t. take the n highest pt particles in each jet
        X_train_particle_transformed = X_train_particle_transformed[:, :args.num_particles, :]
    
    # Compute scale only over real (unmasked) particles to avoid zero-padding deflating std.
    mask_flat = X_train_particle_transformed[:, :, 4].flatten().numpy().astype(bool)
    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())[mask_flat]
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())[mask_flat]
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())[mask_flat]
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())[mask_flat]
    scales = [np.std(e_c), np.std(p_x), np.std(p_y), np.std(p_z)]
    final_scale = np.mean(scales)
    with open(f"{model_output_path}/scale.txt", "w") as f:
        f.write(f"{final_scale}\n")
    X_train_particle_transformed[:, :, :4] = (1/final_scale) * X_train_particle_transformed[:, :, :4]
    print(f"{X_train_particle_transformed[:, :, 0].mean()=} {X_train_particle_transformed[:, :, 1].mean()=} {X_train_particle_transformed[:, :, 2].mean()=} {X_train_particle_transformed[:, :, 3].mean()=}")
    print(f"{X_train_particle_transformed[:, :, 0].std()=} {X_train_particle_transformed[:, :, 1].std()=} {X_train_particle_transformed[:, :, 2].std()=} {X_train_particle_transformed[:, :, 3].std()=}")
    print(f"{X_train_particle_transformed[:, :, 0].max()=} {X_train_particle_transformed[:, :, 1].max()=} {X_train_particle_transformed[:, :, 2].max()=} {X_train_particle_transformed[:, :, 3].max()=}")
    print(f"{X_train_particle_transformed[:, :, 0].min()=} {X_train_particle_transformed[:, :, 1].min()=} {X_train_particle_transformed[:, :, 2].min()=} {X_train_particle_transformed[:, :, 3].min()=}")
    model: LEFTJeN = LEFTJeN(
        max_num_jet_types=NUM_CLASSES,
        max_particles=args.num_particles,
        num_layers=args.n_layers,
        hidden_dim=args.n_hidden,
        use_residual_update=args.use_residual,
        include_pt=True,
    ).to(device)
    
    make_clear_folder(f"{model_output_path}/models")
    torch.save(model.state_dict(), f"{model_output_path}/models/initial_model.pth")
    train_jet_info = X_train[:][1].to(device)
    if args.num_particles < MAX_N_PARTICLES:
        train_jet_info[:, 3] = train_jet_info[:, 3].clamp(max=args.num_particles)
    train_jet_info = train_jet_info[:args.n_train_samples]
    losses = []

    # ── Curriculum: pre-compute bucket assignment for every training sample ───
    # Bucket k ∈ {0, …, N-1}, k=0 sparsest, k=N-1 densest.
    # P(bucket k) ∝ (k+1)^α; high α oversamples dense jets.
    # α decays linearly from curriculum_alpha_start to 0 over num_epochs.
    N_CURRICULUM_BUCKETS = args.n_curriculum_buckets
    n_particles_per_jet = train_jet_info[:, 3].cpu().float()     # (n_train,)
    p_min = n_particles_per_jet.min().item()
    p_max = n_particles_per_jet.max().item()
    bucket_width = (p_max - p_min + 1e-6) / N_CURRICULUM_BUCKETS
    bucket_assignments = ((n_particles_per_jet - p_min) / bucket_width).long()
    bucket_assignments = bucket_assignments.clamp(0, N_CURRICULUM_BUCKETS - 1)
    bucket_counts = torch.bincount(bucket_assignments, minlength=N_CURRICULUM_BUCKETS).float()
    n_nonempty = (bucket_counts > 0).sum().item()
    print(f"Curriculum: {N_CURRICULUM_BUCKETS} buckets, {int(n_nonempty)} non-empty, "
          f"alpha_start={args.curriculum_alpha_start:.2f}")

    # ── ICP cache ─────────────────────────────────────────────────────────────
    x_0_cache: torch.Tensor | None = None
    if args.icp_cache_path is not None:
        print(f"Loading ICP prior cache from {args.icp_cache_path} …")
        with open(args.icp_cache_path, "rb") as f:
            icp_payload = pickle.load(f)
        x_0_cache = torch.from_numpy(icp_payload["x_0_cache"]).float()
        # Validate compatibility
        assert x_0_cache.shape[0] >= len(X_train_particle_transformed), (
            f"ICP cache has {x_0_cache.shape[0]} entries but training set needs "
            f"{len(X_train_particle_transformed)}"
        )
        assert x_0_cache.shape[1] >= args.num_particles, (
            f"ICP cache has {x_0_cache.shape[1]} particles but --num_particles={args.num_particles}"
        )
        x_0_cache = x_0_cache[:len(X_train_particle_transformed), :args.num_particles, :]
        print(f"ICP cache loaded: shape={tuple(x_0_cache.shape)}")

    # ── Time-sampling closure (avoids repeated if/elif inside the inner loop) ─
    def _sample_t(batch_size: int) -> torch.Tensor:
        mode = args.time_sampling if args.use_time_sampling else 'uniform'
        if mode == 'uniform':
            return time_dist(batch_size, device=device, mode='uniform')
        elif mode == 'power_law':
            return time_dist(batch_size, device=device, mode='power_law', a=0.2)
        elif mode == 'lognorm':
            return time_dist(batch_size, device=device, mode='lognorm', mu=-0.5, sigma=1.0)
        else:
            raise ValueError(f"Unknown time_sampling: {mode}")

    lr = 1e-4
    weight_decay = 1e-6
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=lr * 0.1
        )

    epoch_fraction = args.epoch_frac
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))

    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0

        # ── Sample epoch indices (uniform or curriculum) ─────────────────────
        if args.use_curriculum:
            # α decays linearly from α_start (epoch 0) to 0 (epoch num_epochs-1)
            alpha = args.curriculum_alpha_start * (
                1.0 - epoch / max(args.num_epochs - 1, 1)
            )
            # P(bucket k) ∝ (k+1)^α; weight 0 for empty buckets automatically
            # because no samples belong to them.
            bucket_probs = torch.pow(
                torch.arange(1, N_CURRICULUM_BUCKETS + 1, dtype=torch.float), alpha
            )
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_counts_safe = bucket_counts.clamp(min=1)
            sample_weights = (
                bucket_probs[bucket_assignments] / bucket_counts_safe[bucket_assignments]
            )
            # Draw with replacement so the curriculum distribution is exact
            epoch_indices = torch.multinomial(
                sample_weights, samples_per_epoch, replacement=True
            )
        else:
            epoch_indices = torch.randperm(
                len(X_train_particle_transformed)
            )[:samples_per_epoch]

        X_train_epoch = torch.utils.data.Subset(X_train_particle_transformed, epoch_indices)
        train_jet_info_epoch = train_jet_info[epoch_indices]
        x_0_cache_epoch = x_0_cache[epoch_indices] if x_0_cache is not None else None

        paired_dataset = PairedDataset(
            train_jet_info_epoch, X_train_epoch, x_0_cache=x_0_cache_epoch
        )
        train_loader = DataLoader(
            paired_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False
        )

        optimizer.zero_grad()
        # Guard for local testing
        accumulation_steps = min(args.target_batch_size // args.batch_size, args.n_train_samples // args.batch_size - 1)
        accumulated_loss = 0
        total_n_accumulations = 0

        for i, (batch_jet_info, batch_particle_info, batch_x_0_cached) in enumerate(train_loader):

            batch_jet_info = batch_jet_info.to(device)
            batch_particle_info = batch_particle_info.to(device)

            batch_jet_n_particles = batch_jet_info[:, 3]
            batch_jet_pt = batch_jet_info[:, 1]  # normalized pT from FeaturewiseLinear
            encoded_jet_types = jet_attributes.one_hot_enc_jet_type(batch_jet_info[:, 4].long()).to(device)
            # Conditioning vector: [one_hot_type, n_particles, pT]
            batch_jet_info_cropped = torch.cat([
                encoded_jet_types,
                batch_jet_n_particles.unsqueeze(-1),
                batch_jet_pt.unsqueeze(-1),
            ], dim=-1).to(device)

            # Per-sample CFG dropout: each sample independently drops jet conditioning.
            dropout_mask = torch.rand(batch_jet_info_cropped.shape[0], device=device) < args.cfg_null_dropout_rate
            if dropout_mask.any():
                null_vecs = null_vector_like(batch_jet_info_cropped).to(device)
                batch_jet_info_cropped = torch.where(
                    dropout_mask.unsqueeze(-1), null_vecs, batch_jet_info_cropped
                )

            x_1 = batch_particle_info[:, :, :4].to(device)
            true_masks = batch_particle_info[:, :, 4].to(device)   # always use mask

            # TODO: Boost before, can't boost scaled data
            # x_1 = boost_to_com_frame(x_1, mask=true_masks)
            # x_1 = (1/SCALE) * x_1
            if x_0_cache is not None:
                x_0 = batch_x_0_cached.to(device)
            else:
                x_0 = gen_initial_distribution(x_1=x_1).to(device)

            mask_exp = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
            x_1 = mask_exp * x_1
            x_0 = mask_exp * x_0
            
            # mean_std_masked_tensor("x_0", x_0, true_masks)
            # mean_std_masked_tensor("x_1", x_1, true_masks)
            
            t = _sample_t(x_0.shape[0]).to(device)
            t_viewed = t.view(-1, 1, 1)

            """
            The support of the prob dist is convex in polar coordinates
            So we convert x_0 and x_1 to polar
            And calculate x_t in polar
            Then we convert it back to cartesian for the model

            We do loss in cartesian coords as usual - they should be equivalent. The point is to only use valid x_t
            """

            x_t = None
            if args.train_space == 'polar':
                x_0_polar = cartesian_to_EtaPhiPtE(x_0)
                x_1_polar = cartesian_to_EtaPhiPtE(x_1)
                x_t = (1 - (1-args.sigma_min)*t_viewed)*x_0_polar + t_viewed * x_1_polar
                x_t = EtaPhiPtE_to_cartesian(x_t)
            else:
                x_t = (1 - (1-args.sigma_min)*t_viewed)*x_0 + t_viewed * x_1

            x_t = x_t.to(device)
            # mean_std_masked_tensor("x_t", x_t, true_masks)

            conditional_u_t = (x_1 - ((1-args.sigma_min)*x_0)) * mask_exp
            pred = model.forward(x=x_t, t=t, jet_conditions=batch_jet_info_cropped, mask=true_masks)
            pred = pred * mask_exp

            # Loss: mean over real particle-features only (masked slots are zero and excluded)
            n_real = true_masks.sum().clamp(min=1)
            loss = ((conditional_u_t - pred).square() * mask_exp).sum() / (n_real * NUM_PARTICLE_FEATURES)

            loss.backward()
            # Divide by accumulation_steps here so accumulated_loss is the running average,
            # not the sum. This keeps epoch_loss in the same units as the per-batch loss.
            accumulated_loss += loss.item() / accumulation_steps

            if (i + 1) % accumulation_steps == 0:
                grad_stats = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        current_weight = param.data.norm(2).item()
                        grad_norm = param.grad.norm(2).item()
                        grad_mean = param.grad.abs().mean().item()
                        grad_stats[name] = {
                            'norm': grad_norm,
                            'mean': grad_mean,
                            'weight_norm': current_weight,
                            'update_ratio': grad_norm / (current_weight + 1e-8)
                        }
                
                if total_n_accumulations % 10 == 0:
                    with open(f"{model_output_path}/gradient_stats.csv", "a") as f:
                        if epoch == 0 and total_n_accumulations == 0:
                            f.write("epoch,step," + ",".join([f"{name}_grad_norm,{name}_mean,{name}_update_ratio" 
                                                            for name in grad_stats.keys()]) + "\n")
                        row = f"{epoch},{total_n_accumulations},"
                        row += ",".join([f"{s['norm']},{s['mean']},{s['update_ratio']}" 
                                        for s in grad_stats.values()])
                        f.write(row + "\n")

                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= accumulation_steps
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += accumulated_loss  # already averaged over accumulation_steps
                total_n_accumulations += 1
                accumulated_loss = 0

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
                if total_n_accumulations % 10 == 0:
                    print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {(epoch_loss / total_n_accumulations):.4f}")
                
            del x_1, x_0, t, t_viewed, x_t, conditional_u_t, pred, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        losses.append(epoch_loss / total_n_accumulations)
        if args.use_cosine_lr:
            scheduler.step()

    # Logging
    with open(f"{model_output_path}/training_loss.csv", "w") as f:
        f.write("epoch,loss\n")
        for epoch, loss in enumerate(losses):
            f.write(f"{epoch},{loss}\n")

    # Plot per-layer gradient norms over time
    df = pd.read_csv(f"{model_output_path}/gradient_stats.csv")
    grad_fig_out_path = f"{model_output_path}/grad_figures"
    make_clear_folder(grad_fig_out_path)
    layer_names = [col.replace('_norm', '') for col in df.columns if col.endswith('_norm')]
    for layer in layer_names:
        plt.semilogy(df[f'{layer}_norm'], label=layer, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm (log scale)')
        plt.legend()
        plt.savefig(f"{grad_fig_out_path}/{layer}_gradient_norm.png")
        plt.clf()

    torch.save(model.state_dict(), f"{model_output_path}/models/final_model.pth")

    jet_attr_model_loaded = jet_attributes.load_model(model_path=get_model_pth_path(args.output_path)).to(device)

    try:
        make_clear_folder(f"{model_output_path}/vf_viz_cfg")
        make_clear_folder(f"{model_output_path}/vf_viz_nocfg")
        generate_model_vector_field(
            out_dir=f"{model_output_path}/vf_viz_cfg",
            final_model=model,
            jet_attr_model=jet_attr_model_loaded,
            X_test=X_test,
            scale=final_scale,
            n_jet_types=len(args.jet_types),
            n_particles_per_jet=args.num_particles,
            n_features_per_particle=NUM_PARTICLE_FEATURES,
            n_viz_samples=args.n_viz_samples,
            integration_steps=args.integration_steps,
            use_cfg=True,
            cfg_guidance_weight=2.0
        )
        generate_model_vector_field(
            out_dir=f"{model_output_path}/vf_viz_nocfg",
            final_model=model,
            jet_attr_model=jet_attr_model_loaded,
            X_test=X_test,
            scale=final_scale,
            n_jet_types=len(args.jet_types),
            n_particles_per_jet=args.num_particles,
            n_features_per_particle=NUM_PARTICLE_FEATURES,
            n_viz_samples=args.n_viz_samples,
            integration_steps=args.integration_steps,
            use_cfg=False,
        )
    except Exception as e:
        print(f"Error occurred while generating model vector field: {e}")
        with open(f"{model_output_path}/error_log.txt", "a") as f:
            f.write(f"Error occurred while generating model vector field: {e}\n")

    try:
        samples = generate_samples(
            model=model,
            jet_attr_model=jet_attr_model_loaded,
            root_output_path=model_output_path,
            max_particles_per_jet=args.num_particles,
            final_scale=final_scale,
            integration_steps=args.integration_steps,
            n_samples=args.n_samples,
            n_jet_types=len(args.jet_types),
            device=device,
            batch_size=args.batch_size,
            use_cfg=True
        )
    except Exception as e:
        print(f"Error occurred while generating samples: {e}")
        with open(f"{model_output_path}/error_log.txt", "a") as f:
            f.write(f"Error occurred while generating samples: {e}\n")
        exit(1)
    
    try:
        run_save_metrics(
            X_test=X_test,
            jet_types=args.jet_types,
            gen_samples=samples,
            output_path=model_output_path,
            device=device
        )
    except Exception as e:
        print(f"Error occurred while running/saving metrics: {e}")
        with open(f"{model_output_path}/error_log.txt", "a") as f:
            f.write(f"Error occurred while running/saving metrics: {e}\n")