import os
import sys
import json
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import random
from torch.utils.data import DataLoader

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.jet_attributes import NUM_CLASSES
from jet_attr_model import get_model_pth_path
from util.distributions import gen_initial_distribution, time_dist, hyperbolic_interpolant
from util.hyperbolic import pushforward, hyperbolic_loss
from util.coordinates import transform_rel_particle_coordinates_to_cartesian, jacobian_epp_etaphipte
from jetnet.utils import EtaPhiPtE_to_cartesian, cartesian_to_EtaPhiPtE
from util.ema import ModelEMA
from util.file_management import make_clear_folder
from util.viz import generate_model_vector_field
from util.metrics import run_save_metrics
# from util.boost_equiv import boost_to_com_frame
from generate_samples import generate_samples
from data import get_data_path
from cache_icp import canonical_cache_path
from util.mask_helpers import mean_std_masked_tensor
from config import TrainRunConfig, build_config, parse_config_cli, train_config_to_namespace

RANDOM_SEED = 42
MAX_N_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7

# SCALE = 2000

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, jet_info, particle_data, perm_cache=None, rot_cache=None):
        self.jet_info = jet_info
        self.particle_data = particle_data
        self.perm_cache = perm_cache   # optional (N, P) int64 — ICP permutation indices
        self.rot_cache = rot_cache     # optional (N, 3, 3) float32 — ICP rotation matrices

    def __len__(self):
        return len(self.particle_data)

    def __getitem__(self, idx):
        perm = self.perm_cache[idx] if self.perm_cache is not None else torch.zeros(1, dtype=torch.long)
        rot = self.rot_cache[idx] if self.rot_cache is not None else torch.zeros(1)
        return self.jet_info[idx], self.particle_data[idx], perm, rot

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    config_path, overrides = parse_config_cli()
    cfg = build_config(TrainRunConfig, config_path, overrides)
    args = train_config_to_namespace(cfg)

    _is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    args.distributed = args.distributed or _is_torchrun

    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_rank0 = (rank == 0)
    if is_rank0:
        print(f"Using {device} device (world_size={world_size})")

    data_path = get_data_path(args.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(f"{data_path}/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    model_output_path = f"{args.output_path}/train"
    if is_rank0:
        if args.resume_weights:
            os.makedirs(model_output_path, exist_ok=True)
        else:
            make_clear_folder(model_output_path)
            with open(f"{model_output_path}/args.txt", "w") as f:
                f.write("CLI args:\n")
                for arg in vars(args):
                    f.write(f"{arg}: {getattr(args, arg)}\n")
    if args.distributed:
        dist.barrier()  # ranks 1..N-1 wait for rank 0 to create the output dir

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
    if is_rank0:
        with open(f"{model_output_path}/scale.txt", "w") as f:
            f.write(f"{final_scale}\n")
    X_train_particle_transformed[:, :, :4] = (1/final_scale) * X_train_particle_transformed[:, :, :4]
    if is_rank0:
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
        use_reference_vectors=args.use_reference_vectors,
        use_node_scalars=args.use_node_scalars,
        use_adaln=args.use_adaln,
        use_attention=args.use_attention,
    ).to(device)
    
    start_epoch = 0
    losses = []
    if args.resume_weights:
        checkpoint = torch.load(args.resume_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        losses = checkpoint.get("losses", [])
        if is_rank0:
            print(f"Resumed from checkpoint at epoch {start_epoch - 1}; "
                  f"running {args.num_epochs} more epochs ({start_epoch}→{start_epoch + args.num_epochs - 1}).")

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if args.distributed else model

    # EMA of weights (Phase 2.1). Shadow starts from the (possibly resumed) weights.
    ema = None
    if args.use_ema:
        ema = ModelEMA(raw_model, decay=args.ema_decay)
        if args.resume_weights and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"], device=device)
            if is_rank0:
                print("Resumed EMA shadow weights from checkpoint.")

    # Self-describing model config, embedded in every checkpoint so a run can be resumed or
    # loaded for inference without re-specifying the architecture flags on the CLI.
    run_config = {
        "num_particles": args.num_particles,
        "n_layers": args.n_layers,
        "n_hidden": args.n_hidden,
        "use_residual": args.use_residual,
        "include_pt": True,
        "use_reference_vectors": args.use_reference_vectors,
        "use_node_scalars": args.use_node_scalars,
        "use_adaln": args.use_adaln,
        "use_attention": args.use_attention,
        "jet_types": args.jet_types,
        "final_scale": float(final_scale),
    }
    # Full config (all sections), embedded alongside the legacy architecture-only
    # `config` dict so infer.py can auto-load every knob a run was trained with.
    full_config = cfg.model_dump()
    if args.resume_weights and is_rank0:
        prev = checkpoint.get("config")
        if prev is not None:
            mism = {k: (prev.get(k), run_config.get(k))
                    for k in ("n_layers", "n_hidden", "num_particles", "use_reference_vectors",
                              "use_node_scalars", "use_adaln", "use_attention")
                    if prev.get(k) != run_config.get(k)}
            if mism:
                print(f"WARNING: resume architecture flags differ from checkpoint: {mism}. "
                      f"Re-run with matching flags or the loaded weights are wrong.")

    if not args.resume_weights:
        if is_rank0:
            make_clear_folder(f"{model_output_path}/models")
        if args.distributed:
            dist.barrier()
    else:
        if is_rank0:
            os.makedirs(f"{model_output_path}/models", exist_ok=True)

    train_jet_info = X_train[:][1].to(device)
    if args.num_particles < MAX_N_PARTICLES:
        train_jet_info[:, 3] = train_jet_info[:, 3].clamp(max=args.num_particles)
    train_jet_info = train_jet_info[:args.n_train_samples]

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
    if is_rank0:
        print(f"Curriculum: {N_CURRICULUM_BUCKETS} buckets, {int(n_nonempty)} non-empty, "
              f"alpha_start={args.curriculum_alpha_start:.2f}")

    # ── ICP cache ─────────────────────────────────────────────────────────────
    perm_cache: torch.Tensor | None = None
    rot_cache: torch.Tensor | None = None
    icp_cache_path = args.icp_cache_path
    if icp_cache_path is None:
        auto = canonical_cache_path(args.cache_dir, args.jet_types, args.num_particles)
        if os.path.exists(auto):
            logging.info(f"Auto-discovered ICP cache: {auto}")
            icp_cache_path = auto
    if icp_cache_path is not None:
        if is_rank0:
            print(f"Loading ICP cache from {icp_cache_path} …")
        with open(icp_cache_path, "rb") as f:
            icp_payload = pickle.load(f)
        if "perm_cache" not in icp_payload:
            raise ValueError(
                f"ICP cache at '{icp_cache_path}' is old format (x_0_cache key). "
                "Re-run cache_icp.py — it will auto-detect, delete, and recompute the cache."
            )
        n_train = len(X_train_particle_transformed)
        perm_cache = torch.from_numpy(icp_payload["perm_cache"]).long()
        assert perm_cache.shape[0] >= n_train, (
            f"ICP cache has {perm_cache.shape[0]} entries but training set needs {n_train}"
        )
        assert perm_cache.shape[1] >= args.num_particles, (
            f"ICP cache has {perm_cache.shape[1]} particles but --num_particles={args.num_particles}"
        )
        perm_cache = perm_cache[:n_train, :args.num_particles]
        if "rot_cache" in icp_payload:
            rot_cache = torch.from_numpy(icp_payload["rot_cache"]).float()
            rot_cache = rot_cache[:n_train]
        if is_rank0:
            print(f"ICP cache loaded: perm={tuple(perm_cache.shape)}"
                  + (f"  rot={tuple(rot_cache.shape)}" if rot_cache is not None else "  (no rot)"))

   
    def _sample_t(batch_size: int) -> torch.Tensor:
        mode = args.time_sampling if args.use_time_sampling else 'uniform'
        if mode == 'uniform':
            return time_dist(batch_size, device=device, mode='uniform')
        elif mode == 'power_law':
            return time_dist(batch_size, device=device, mode='power_law', a=-0.2)
        elif mode == 'lognorm':
            return time_dist(batch_size, device=device, mode='lognorm', mu=-0.5, sigma=1.0)
        else:
            raise ValueError(f"Unknown time_sampling: {mode}")

    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.use_cosine_lr:
        t0 = args.lr_t0 if args.lr_t0 > 0 else (args.num_epochs // 4) if args.num_epochs >= 20 else max(1, args.num_epochs // 2)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=1, eta_min=lr * args.eta_min_factor
        )
        if args.lr_warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0,
                total_iters=args.lr_warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[args.lr_warmup_epochs]
            )
        else:
            scheduler = cosine_sched

    if args.resume_weights:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.use_cosine_lr and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch_fraction = args.epoch_frac
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))

    
    total_epochs = start_epoch + args.num_epochs
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0
        num_batches = 0

        # ── Sample epoch indices (uniform or curriculum) ─────────────────────
        # All ranks produce the same indices via a deterministic seed, then each
        # takes its own contiguous slice so every rank sees a diverse bucket mix.
        _epoch_seed = RANDOM_SEED + epoch * 1000
        _rng = torch.get_rng_state()
        torch.manual_seed(_epoch_seed)

        if args.use_curriculum:
            # alpha decays linearly from alpha_start (epoch 0) to 0 (final epoch),
            # using total_epochs so the schedule is continuous across resume boundaries.
            alpha = args.curriculum_alpha_start * (
                1.0 - epoch / max(total_epochs - 1, 1)
            )
            # P(bucket k) \propto (k+1)^α; weight 0 for empty buckets automatically
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

        torch.set_rng_state(_rng)  # restore so training noise is unaffected

        if args.distributed:
            shard_size = len(epoch_indices) // world_size
            epoch_indices = epoch_indices[:shard_size * world_size]
            epoch_indices = epoch_indices[rank::world_size]  # stride split → diverse bucket mix

        X_train_epoch = torch.utils.data.Subset(X_train_particle_transformed, epoch_indices)
        train_jet_info_epoch = train_jet_info[epoch_indices]
        perm_cache_epoch = perm_cache[epoch_indices] if perm_cache is not None else None
        rot_cache_epoch = rot_cache[epoch_indices] if rot_cache is not None else None

        paired_dataset = PairedDataset(
            train_jet_info_epoch, X_train_epoch, perm_cache=perm_cache_epoch, rot_cache=rot_cache_epoch
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

        for i, (batch_jet_info, batch_particle_info, batch_perm_cached, batch_rot_cached) in enumerate(train_loader):

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

            # Per-sample CFG dropout: each sample independently drops jet type + pT
            # conditioning. n_particles is preserved because it is already encoded in
            # the particle mask, so nulling it adds no guidance signal and only weakens
            # the unconditional-conditional gap.
            dropout_mask = torch.rand(batch_jet_info_cropped.shape[0], device=device) < args.cfg_null_dropout_rate
            if dropout_mask.any():
                null_for_batch = raw_model.make_null_cond(batch_jet_info_cropped)
                batch_jet_info_cropped = torch.where(
                    dropout_mask.unsqueeze(-1), null_for_batch, batch_jet_info_cropped
                )

            x_1 = batch_particle_info[:, :, :4].to(device)
            true_masks = batch_particle_info[:, :, 4].to(device)   # always use mask

            # TODO: Boost before, can't boost scaled data
            # x_1 = boost_to_com_frame(x_1, mask=true_masks)
            # x_1 = (1/SCALE) * x_1
            x_0 = gen_initial_distribution(
                x_1=x_1, prior_dist=args.prior_dist, jet_features=batch_jet_info, device=device,
            ).to(device)
            if perm_cache is not None:
                # Apply the ICP-derived permutation then rotation to the fresh prior sample.
                # Permutation: gather along particle dim using (B, P, 4) index.
                batch_perm = batch_perm_cached.to(device)
                x_0 = torch.gather(x_0, 1, batch_perm.unsqueeze(-1).expand(-1, -1, x_0.shape[-1]))
                if rot_cache is not None:
                    # Rotate 3-momenta (indices 1:4): (B, P, 3) @ (B, 3, 3)^T
                    batch_rot = batch_rot_cached.to(device)
                    x_0 = torch.cat([
                        x_0[:, :, :1],
                        torch.bmm(x_0[:, :, 1:4], batch_rot.transpose(1, 2)),
                    ], dim=-1)

            mask_exp = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
            x_1 = mask_exp * x_1
            x_0 = mask_exp * x_0

            # Reference virtual particles: e_t=(1,0,0,0) and the true jet 4-momentum
            # (sum of the target constituents, in scaled space). None when disabled.
            ref_vectors = None
            if args.use_reference_vectors:
                e_t = torch.zeros(x_1.shape[0], 1, 4, device=device, dtype=x_1.dtype)
                e_t[..., 0] = 1.0
                jet_p4 = (x_1 * true_masks.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B, 1, 4)
                ref_vectors = torch.cat([e_t, jet_p4], dim=1)  # (B, 2, 4)


            # mean_std_masked_tensor("x_0", x_0, true_masks)
            # mean_std_masked_tensor("x_1", x_1, true_masks)
            
            t = _sample_t(x_0.shape[0]).to(device)
            t_viewed = t.view(-1, 1, 1)

            if args.use_hyperbolic:
                # Riemannian flow matching: geodesic interpolant in the Poincaré ball.
                # x_t is returned in Cartesian for the model; y_t and u_t_ball stay in the ball.
                x_t, y_t, u_t_ball = hyperbolic_interpolant(x_0, x_1, t)
                x_t = x_t.to(device)
                y_t = y_t.to(device)
                u_t_ball = u_t_ball * mask_exp   # zero out padding in target

                pred = model.forward(x=x_t, t=t, jet_conditions=batch_jet_info_cropped, mask=true_masks, ref_vectors=ref_vectors)
                pred = pred * mask_exp

                # Push Cartesian model output into the ball tangent space, then compute
                # the Riemannian loss ||v_theta - u_t||^2_g (Equation 14).
                pred_ball = pushforward(x_t, pred) * mask_exp
                loss = hyperbolic_loss(pred_ball, u_t_ball, y_t, mask=true_masks)
            else:
                # Euclidean flow matching (polar or Cartesian interpolation).
                if args.train_space == 'polar':
                    x_0_polar = cartesian_to_EtaPhiPtE(x_0)
                    x_1_polar = cartesian_to_EtaPhiPtE(x_1)
                    x_t = (1 - (1-args.sigma_min)*t_viewed)*x_0_polar + t_viewed * x_1_polar
                    x_t = EtaPhiPtE_to_cartesian(x_t)
                else:
                    x_t = (1 - (1-args.sigma_min)*t_viewed)*x_0 + t_viewed * x_1
                x_t = x_t.to(device)

                conditional_u_t = (x_1 - ((1-args.sigma_min)*x_0)) * mask_exp
                pred = model.forward(x=x_t, t=t, jet_conditions=batch_jet_info_cropped, mask=true_masks, ref_vectors=ref_vectors)
                pred = pred * mask_exp

                n_real = true_masks.sum().clamp(min=1)
                loss = ((conditional_u_t - pred).square() * mask_exp).sum() / (n_real * NUM_PARTICLE_FEATURES)

            is_last_accum_step = ((i + 1) % accumulation_steps == 0)
            ctx = model.no_sync() if (args.distributed and not is_last_accum_step) else nullcontext()
            with ctx:
                loss.backward()
            # Divide by accumulation_steps here so accumulated_loss is the running average,
            # not the sum. This keeps epoch_loss in the same units as the per-batch loss.
            accumulated_loss += loss.item() / accumulation_steps

            if (i + 1) % accumulation_steps == 0:
                grad_stats = {}
                for name, param in raw_model.named_parameters():
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
                
                if is_rank0 and total_n_accumulations % 10 == 0:
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
                if ema is not None:
                    ema.update(raw_model)

                epoch_loss += accumulated_loss  # already averaged over accumulation_steps
                total_n_accumulations += 1
                accumulated_loss = 0

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
                if is_rank0 and total_n_accumulations % 10 == 0:
                    print(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {(epoch_loss / total_n_accumulations):.4f}")
                
            del x_1, x_0, t, x_t, pred, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.distributed:
            loss_tensor = torch.tensor(epoch_loss / total_n_accumulations, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            epoch_mean_loss = loss_tensor.item()
        else:
            epoch_mean_loss = epoch_loss / total_n_accumulations
        losses.append(epoch_mean_loss)

        # Overwrite latest checkpoint so training can be resumed at any point.
        if is_rank0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "config": run_config,
                "full_config": full_config,
            }
            if args.use_cosine_lr:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if ema is not None:
                ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt, f"{model_output_path}/models/latest_checkpoint.pth")

        if args.use_cosine_lr:
            scheduler.step()

    if args.distributed:
        dist.barrier()          # all ranks finish training before rank 0 does inference
        dist.destroy_process_group()

    if is_rank0:
        # Logging — on resume, append only the newly-completed epochs.
        write_mode = "a" if args.resume_weights else "w"
        with open(f"{model_output_path}/training_loss.csv", write_mode) as f:
            if not args.resume_weights:
                f.write("epoch,loss\n")
            for epoch_i, loss_val in enumerate(losses[start_epoch:], start=start_epoch):
                f.write(f"{epoch_i},{loss_val}\n")

        torch.save(raw_model.state_dict(), f"{model_output_path}/models/final_model.pth")

        # Complete resume point: everything needed to continue training from the final model
        # (raw weights, optimizer, scheduler, EMA shadow, epoch, losses, self-describing config).
        final_ckpt = {
            "epoch": total_epochs - 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "config": run_config,
            "full_config": full_config,
        }
        if args.use_cosine_lr:
            final_ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if ema is not None:
            final_ckpt["ema_state_dict"] = ema.state_dict()
        torch.save(final_ckpt, f"{model_output_path}/models/final_checkpoint.pth")

        # If EMA is active, save the shadow and use it for all downstream sampling/metrics.
        if ema is not None:
            ema.copy_to(raw_model)
            torch.save(raw_model.state_dict(), f"{model_output_path}/models/ema_model.pth")
            print("Using EMA weights for sample generation and metrics.")

        jet_attr_model_loaded = jet_attributes.load_model(model_path=get_model_pth_path(args.output_path)).to(device)

        try:
            make_clear_folder(f"{model_output_path}/vf_viz_cfg")
            make_clear_folder(f"{model_output_path}/vf_viz_nocfg")
            generate_model_vector_field(
                out_dir=f"{model_output_path}/vf_viz_cfg",
                final_model=raw_model,
                jet_attr_model=jet_attr_model_loaded,
                X_test=X_test,
                scale=final_scale,
                n_jet_types=len(args.jet_types),
                n_particles_per_jet=args.num_particles,
                n_features_per_particle=NUM_PARTICLE_FEATURES,
                # set to 100 viz samples if 150 particle jets
                n_viz_samples=args.n_viz_samples if args.num_particles < MAX_N_PARTICLES else 100,
                integration_steps=args.integration_steps,
                use_cfg=True,
                cfg_guidance_weight=2.0,
                use_hyperbolic=args.use_hyperbolic,
            )
            generate_model_vector_field(
                out_dir=f"{model_output_path}/vf_viz_nocfg",
                final_model=raw_model,
                jet_attr_model=jet_attr_model_loaded,
                X_test=X_test,
                scale=final_scale,
                n_jet_types=len(args.jet_types),
                n_particles_per_jet=args.num_particles,
                n_features_per_particle=NUM_PARTICLE_FEATURES,
                n_viz_samples=args.n_viz_samples if args.num_particles < MAX_N_PARTICLES else 100,
                integration_steps=args.integration_steps,
                use_cfg=False,
                use_hyperbolic=args.use_hyperbolic,
            )
        except Exception as e:
            print(f"Error occurred while generating model vector field: {e}")
            with open(f"{model_output_path}/error_log.txt", "a") as f:
                f.write(f"Error occurred while generating model vector field: {e}\n")

        try:
            samples = generate_samples(
                model=raw_model,
                jet_attr_model=jet_attr_model_loaded,
                root_output_path=model_output_path,
                max_particles_per_jet=args.num_particles,
                final_scale=final_scale,
                integration_steps=args.integration_steps,
                n_samples=args.n_samples,
                n_jet_types=len(args.jet_types),
                device=device,
                batch_size=args.batch_size if args.num_particles < MAX_N_PARTICLES else 16,
                use_cfg=False,
                use_hyperbolic=args.use_hyperbolic,
                use_reference_vectors=args.use_reference_vectors,
            )
        except Exception as e:
            print(f"Error occurred while generating samples: {e}")
            with open(f"{model_output_path}/error_log.txt", "a") as f:
                f.write(f"Error occurred while generating samples: {e}\n")
            exit(1)

        # Persist a small sample subset so plots/metrics can be regenerated without re-running.
        try:
            torch.save(samples[:10000].cpu(), f"{model_output_path}/samples_subset.pt")
        except Exception as e:
            print(f"Error saving sample subset: {e}")

        eval_info = {}
        try:
            eval_info = run_save_metrics(
                X_test=X_test,
                jet_types=args.jet_types,
                gen_samples=samples,
                output_path=model_output_path,
                device=device
            ) or {}
        except Exception as e:
            print(f"Error occurred while running/saving metrics: {e}")
            with open(f"{model_output_path}/error_log.txt", "a") as f:
                f.write(f"Error occurred while running/saving metrics: {e}\n")

        # Compact run summary for at-a-glance comparison across runs.
        try:
            git_commit = None
            git_commit_path = f"{args.output_path}/git_commit.txt"
            if os.path.exists(git_commit_path):
                with open(git_commit_path) as gf:
                    git_commit = gf.read().strip()
            summary = {
                "final_loss": losses[-1] if losses else None,
                "num_epochs": total_epochs,
                "git_commit": git_commit,
                "config": run_config,
                "full_config": full_config,
                "metrics": {k: eval_info.get(k) for k in (
                    "w1m", "w1p", "w1efp", "fpd",
                    "frac_negative_energy", "frac_spacelike", "msq_median",
                )},
            }
            with open(f"{model_output_path}/summary.json", "w") as f:
                json.dump(summary, f, indent=2,
                          default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
        except Exception as e:
            print(f"Error writing summary.json: {e}")