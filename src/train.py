import os
import json
import time
import pickle
import logging
import math

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
from util.distributions import gen_initial_distribution, time_dist
from util.coordinates import (deterministic_jet_phi,
                              transform_rel_particle_coordinates_to_cartesian)
from util.ema import ModelEMA
from util.file_management import make_clear_folder
from util.viz import generate_model_vector_field
from util.metrics import run_save_metrics
# from util.boost_equiv import boost_to_com_frame
from generate_samples import generate_samples
from data import get_data_path
from cache_icp import (resolve_training_cache_path, source_dataset_fingerprint,
                       validate_cache_metadata)
from util.qualification import loss_improvement_summary, optimizer_limit_reached
from util.rng import capture_rng_state, keyed_seed, keyed_torch_rng, restore_rng_state
from util.lr_schedule import build_epoch_scheduler, build_step_scheduler
from training import flow_matching_loss
from config import (TrainRunConfig, build_config, parse_config_cli, train_config_to_namespace,
                    generation_controls_from_namespace)

RANDOM_SEED = 42
MAX_N_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7

# SCALE = 2000

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, jet_info, particle_data, jet_phi, paired_x0=None):
        self.jet_info = jet_info
        self.particle_data = particle_data
        self.jet_phi = jet_phi
        self.paired_x0 = paired_x0

    def __len__(self):
        return len(self.particle_data)

    def __getitem__(self, idx):
        x0 = self.paired_x0[idx] if self.paired_x0 is not None else torch.zeros(1)
        return self.jet_info[idx], self.particle_data[idx], self.jet_phi[idx], x0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    config_path, overrides = parse_config_cli()
    cfg = build_config(TrainRunConfig, config_path, overrides)
    args = train_config_to_namespace(cfg)

    _is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    torchrun_world_size = int(os.environ["WORLD_SIZE"]) if _is_torchrun else 1
    # torchrun is also our single-GPU launcher. Do not construct a one-rank DDP
    # process group: it adds synchronization/unused-parameter overhead without
    # changing the optimization.
    args.distributed = args.distributed or torchrun_world_size > 1

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        # Bind NCCL to the process-local GPU up front. Initializing first leaves
        # the rank-to-device mapping unknown and can hang at the first barrier.
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
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

    train_jet_phi = deterministic_jet_phi(len(X_train), RANDOM_SEED)
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(
        X_train, jet_phi=train_jet_phi).to('cpu')
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
    # Model initialization is an isolated treatment stream. Its parameter count can no
    # longer perturb data order, time, dropout, or prior draws in another arm.
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    model: LEFTJeN = LEFTJeN(
        max_num_jet_types=NUM_CLASSES,
        max_particles=args.num_particles,
        num_layers=args.n_layers,
        hidden_dim=args.n_hidden,
        use_residual_update=args.use_residual,
        include_pt=True,
        use_reference_vectors=args.use_reference_vectors,
        use_node_scalars=args.use_node_scalars,
        node_scalar_seed=args.node_scalar_seed,
        use_adaln=args.use_adaln,
        use_attention=args.use_attention,
        backbone=args.backbone,
        include_mass_condition=args.include_mass_condition,
        num_attention_heads=args.num_attention_heads,
        vector_channels=args.vector_channels,
        regulator_mass=args.regulator_mass,
        velocity_readout_init=args.velocity_readout_init,
    ).to(device)
    
    start_epoch = 0
    resume_minibatch = 0
    losses = []
    global_optimizer_step = 0
    if args.resume_weights:
        checkpoint = torch.load(args.resume_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("resume_epoch", checkpoint["epoch"] + 1))
        resume_minibatch = int(checkpoint.get("resume_minibatch", 0))
        losses = checkpoint.get("losses", [])
        global_optimizer_step = int(checkpoint.get("global_optimizer_step", 0))
        if is_rank0:
            print(f"Resumed from checkpoint at epoch {start_epoch - 1}; "
                  f"running {args.num_epochs} more epochs ({start_epoch}→{start_epoch + args.num_epochs - 1}).")
    if args.distributed:
        # find_unused_parameters=True: the final layer's global/node-update sub-branch
        # (phi_m, phi_g, global_sf, alpha, +phi_h) produces g_new/h_new that are
        # discarded (velocity = x_particles - x0 uses only the coordinate stream), so
        # those params get no gradient. Also covers the rare zero-dropout CFG batch
        # where null_cond is not exercised. Output math is unchanged.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
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
        "backbone": args.backbone,
        "include_mass_condition": args.include_mass_condition,
        "num_attention_heads": args.num_attention_heads,
        "vector_channels": args.vector_channels,
        "velocity_readout_init": args.velocity_readout_init,
        "regulator_mass": args.regulator_mass,
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
                              "use_node_scalars", "use_adaln", "use_attention", "backbone",
                              "include_mass_condition", "num_attention_heads", "vector_channels",
                              "velocity_readout_init")
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
    paired_x0_cache: torch.Tensor | None = None
    icp_cache_path = resolve_training_cache_path(
        args.use_icp, args.icp_cache_path, args.cache_dir, args.jet_types, args.num_particles
    )
    if args.use_icp and icp_cache_path is not None:
        if is_rank0:
            print(f"Loading ICP cache from {icp_cache_path} …")
        with open(icp_cache_path, "rb") as f:
            icp_payload = pickle.load(f)
        n_train = len(X_train_particle_transformed)
        expected = {"source_dataset_fingerprint": source_dataset_fingerprint(
                        X_train, n_train, args.num_particles),
                    "dataset_indices": list(range(n_train)), "jet_types": list(args.jet_types),
                    "prior_dist": args.prior_dist, "seed": RANDOM_SEED,
                    "jet_phi_convention": "index_seeded_v1",
                    "jet_phi_seed": RANDOM_SEED,
                    "num_particles": args.num_particles, "final_scale": float(final_scale),
                    "geometry": ("mass_shell" if args.use_hyperbolic and
                                 args.hyperbolic_model == "mass_shell" else "euclidean")}
        if expected["geometry"] == "mass_shell":
            expected["regulator_mass"] = args.regulator_mass
            expected["assignment_cost"] = args.icp_assignment_cost
        validate_cache_metadata(icp_payload, expected)
        paired_x0_cache = torch.from_numpy(icp_payload["paired_x0"]).float()
        paired_x0_cache = paired_x0_cache[:n_train, :args.num_particles]
        if is_rank0:
            print(f"ICP paired x_0 cache loaded: {tuple(paired_x0_cache.shape)}")

   
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

    epoch_fraction = args.epoch_frac
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))
    local_samples_per_epoch = (
        samples_per_epoch // world_size if args.distributed else samples_per_epoch
    )
    accumulation_steps = args.target_batch_size // args.batch_size
    minibatches_per_epoch = math.ceil(local_samples_per_epoch / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(minibatches_per_epoch / accumulation_steps)

    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    planned_optimizer_steps = (
        args.max_optimizer_steps
        if args.max_optimizer_steps is not None
        else args.num_epochs * optimizer_steps_per_epoch
    )
    schedule_definition = {
        "lr_step_unit": args.lr_step_unit,
        "planned_optimizer_steps": planned_optimizer_steps,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "warmup_epochs": args.lr_warmup_epochs,
        "eta_min_factor": args.eta_min_factor,
        "schedule": args.lr_schedule,
        "restart_epoch": args.lr_t0,
    }
    if args.resume_weights and checkpoint.get("schedule_definition") is not None:
        previous_schedule = checkpoint["schedule_definition"]
        if previous_schedule != schedule_definition:
            raise ValueError(
                "resume schedule definition differs from checkpoint: "
                f"checkpoint={previous_schedule}, requested={schedule_definition}"
            )

    if args.use_cosine_lr:
        if args.lr_step_unit == "optimizer_step":
            scheduler = build_step_scheduler(
                optimizer,
                total_steps=planned_optimizer_steps,
                steps_per_epoch=optimizer_steps_per_epoch,
                warmup_epochs=args.lr_warmup_epochs,
                eta_min_factor=args.eta_min_factor,
                schedule=args.lr_schedule,
                restart_epoch=args.lr_t0,
            )
        else:
            scheduler = build_epoch_scheduler(
                optimizer,
                total_epochs=args.num_epochs,
                warmup_epochs=args.lr_warmup_epochs,
                eta_min_factor=args.eta_min_factor,
                schedule=args.lr_schedule,
                restart_epoch=args.lr_t0,
            )

    if args.resume_weights:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.use_cosine_lr and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))

    probe_steps = set(args.stability_probe_steps)
    probe_jet_attr_model = None

    def run_stability_probe(optimizer_step: int, epoch: int | None = None,
                            minibatch: int | None = None) -> None:
        """Run a deterministic EMA sampler probe without perturbing training state."""
        global probe_jet_attr_model
        if optimizer_step not in probe_steps:
            return
        if args.distributed:
            dist.barrier()
        if is_rank0:
            probe_dir = f"{model_output_path}/stability_probes/step_{optimizer_step:06d}"
            os.makedirs(probe_dir, exist_ok=True)
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            raw_state = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
            was_training = raw_model.training
            try:
                if args.stability_probe_save_checkpoints:
                    probe_at_epoch_end = (minibatch is not None
                                          and minibatch + 1 >= len(train_loader))
                    probe_ckpt = {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "losses": losses,
                        "global_optimizer_step": optimizer_step,
                        "config": run_config,
                        "full_config": full_config,
                        "rng_state": capture_rng_state(),
                        "schedule_definition": schedule_definition,
                        "resume_epoch": max(0, (epoch + 1 if probe_at_epoch_end else epoch)
                                            if epoch is not None else 0),
                        "resume_minibatch": (0 if probe_at_epoch_end else
                                             minibatch + 1 if minibatch is not None else 0),
                    }
                    if args.use_cosine_lr:
                        probe_ckpt["scheduler_state_dict"] = scheduler.state_dict()
                    if ema is not None:
                        probe_ckpt["ema_state_dict"] = ema.state_dict()
                    torch.save(probe_ckpt, f"{probe_dir}/checkpoint.pth")
                if ema is not None:
                    ema.copy_to(raw_model)
                random.seed(args.inference_seed)
                np.random.seed(args.inference_seed)
                torch.manual_seed(args.inference_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.inference_seed)
                if probe_jet_attr_model is None:
                    probe_jet_attr_model = jet_attributes.load_model(
                        model_path=get_model_pth_path(args.output_path)
                    ).to(device)
                probe_outputs = generate_samples(
                    model=raw_model,
                    jet_attr_model=probe_jet_attr_model,
                    root_output_path=probe_dir,
                    max_particles_per_jet=args.num_particles,
                    final_scale=final_scale,
                    integration_steps=args.stability_probe_integration_steps,
                    n_samples=args.stability_probe_samples,
                    n_jet_types=len(args.jet_types),
                    device=device,
                    batch_size=min(args.batch_size, args.stability_probe_samples),
                    **generation_controls_from_namespace(args),
                )
                del probe_outputs
                diagnostics_path = f"{probe_dir}/generation_diagnostics.json"
                with open(diagnostics_path) as handle:
                    diagnostics = json.load(handle)
                n_unstable = int(diagnostics.get("n_unstable", diagnostics["n_failed"]))
                summary = {
                    "optimizer_step": optimizer_step,
                    "velocity_readout_init": args.velocity_readout_init,
                    "n_total": int(diagnostics["n_total"]),
                    "n_unstable": n_unstable,
                    "invalid_fraction": n_unstable / max(int(diagnostics["n_total"]), 1),
                    "n_crossed_max_abs_1e6": int(diagnostics["n_crossed_max_abs_1e6"]),
                    "failure_reason_counts": diagnostics.get("failure_reason_counts", {}),
                    "integration_telemetry_quantiles": diagnostics.get(
                        "integration_telemetry_quantiles", {}
                    ),
                }
                with open(f"{probe_dir}/probe_summary.json", "w") as handle:
                    json.dump(summary, handle, indent=2)
                print(
                    f"Stability probe step={optimizer_step}: "
                    f"unstable={n_unstable}/{summary['n_total']} "
                    f"reasons={summary['failure_reason_counts']}"
                )
            finally:
                raw_model.load_state_dict(raw_state, strict=True)
                raw_model.train(was_training)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
        if args.distributed:
            dist.barrier()

    run_stability_probe(global_optimizer_step, start_epoch - 1)

    
    total_epochs = start_epoch + args.num_epochs
    train_start_time = time.time()
    reached_optimizer_limit = optimizer_limit_reached(
        global_optimizer_step, args.max_optimizer_steps
    )
    last_completed_epoch = start_epoch - 1
    next_resume_epoch = start_epoch
    next_resume_minibatch = resume_minibatch
    for epoch in range(start_epoch, total_epochs):
        if reached_optimizer_limit:
            break
        epoch_loss = 0
        epoch_optimizer_steps = 0

        # ── Sample epoch indices (uniform or curriculum) ─────────────────────
        # All ranks produce the same indices via a deterministic seed, then each
        # takes its own contiguous slice so every rank sees a diverse bucket mix.
        selection_generator = torch.Generator(device="cpu")
        selection_generator.manual_seed(keyed_seed(args.data_order_seed, epoch, 0, 0))

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
                sample_weights, samples_per_epoch, replacement=True,
                generator=selection_generator,
            )
        else:
            epoch_indices = torch.randperm(
                len(X_train_particle_transformed), generator=selection_generator,
            )[:samples_per_epoch]

        if args.distributed:
            shard_size = len(epoch_indices) // world_size
            epoch_indices = epoch_indices[:shard_size * world_size]
            epoch_indices = epoch_indices[rank::world_size]  # stride split → diverse bucket mix

        X_train_epoch = torch.utils.data.Subset(X_train_particle_transformed, epoch_indices)
        train_jet_info_epoch = train_jet_info[epoch_indices]
        train_jet_phi_epoch = train_jet_phi[epoch_indices]
        paired_x0_epoch = paired_x0_cache[epoch_indices] if paired_x0_cache is not None else None

        paired_dataset = PairedDataset(
            train_jet_info_epoch, X_train_epoch, train_jet_phi_epoch, paired_x0=paired_x0_epoch
        )
        train_loader = DataLoader(
            paired_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            generator=torch.Generator(device="cpu").manual_seed(
                keyed_seed(args.data_order_seed, epoch, 1, rank)
            ),
        )

        optimizer.zero_grad()
        accumulation_steps = args.target_batch_size // args.batch_size
        accumulated_loss = 0
        accumulated_batches = 0

        for i, (batch_jet_info, batch_particle_info, batch_jet_phi, batch_x0_cached) in enumerate(train_loader):
            if epoch == start_epoch and i < resume_minibatch:
                continue

            batch_jet_info = batch_jet_info.to(device)
            batch_particle_info = batch_particle_info.to(device)

            batch_jet_n_particles = batch_jet_info[:, 3]
            batch_jet_pt = batch_jet_info[:, 1]  # normalized pT from FeaturewiseLinear
            batch_jet_mass = batch_jet_info[:, 2]
            encoded_jet_types = jet_attributes.one_hot_enc_jet_type(batch_jet_info[:, 4].long()).to(device)
            # Legacy checkpoints consume raw pT. v2 uses model-scaled pT and mass.
            cond_pt = (batch_jet_pt / final_scale
                       if args.backbone == "tangent_attention" else batch_jet_pt)
            condition_parts = [
                encoded_jet_types,
                batch_jet_n_particles.unsqueeze(-1),
                cond_pt.unsqueeze(-1),
            ]
            if args.include_mass_condition:
                condition_parts.append((batch_jet_mass / final_scale).unsqueeze(-1))
            batch_jet_info_cropped = torch.cat(condition_parts, dim=-1).to(device)

            # Per-sample CFG dropout: each sample independently drops jet type + pT
            # conditioning. n_particles is preserved because it is already encoded in
            # the particle mask, so nulling it adds no guidance signal and only weakens
            # the unconditional-conditional gap.
            with keyed_torch_rng(args.dropout_seed, epoch, i, rank, device):
                dropout_mask = (torch.rand(batch_jet_info_cropped.shape[0], device=device)
                                < args.cfg_null_dropout_rate)
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
            if paired_x0_cache is not None:
                x_0 = batch_x0_cached.to(device)
            else:
                with keyed_torch_rng(args.prior_seed, epoch, i, rank, device):
                    x_0 = gen_initial_distribution(
                        x_1=x_1, prior_dist=args.prior_dist,
                        jet_features=batch_jet_info, jet_phi=batch_jet_phi.to(device),
                        device=device, model_scale=final_scale,
                    ).to(device)

            mask_exp = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
            x_1 = mask_exp * x_1
            x_0 = mask_exp * x_0

            # External physical massless conditioning jet.  This is reproducible from the
            # same attributes at inference and never leaks the target constituent sum.
            ref_vectors = None
            if args.use_reference_vectors:
                from util.coordinates import build_reference_vectors
                ref_vectors = build_reference_vectors(batch_jet_info[:, 0], batch_jet_info[:, 1],
                                                      final_scale, device,
                                                      jet_phi=batch_jet_phi.to(device),
                                                      jet_mass=(batch_jet_info[:, 2]
                                                                if args.include_mass_condition else None))

            with keyed_torch_rng(args.time_seed, epoch, i, rank, device):
                t = _sample_t(x_0.shape[0]).to(device)
            loss = flow_matching_loss(
                model=model, raw_model=raw_model, config=args,
                x0=x_0, x1=x_1, t=t, mask=true_masks,
                conditions=batch_jet_info_cropped, references=ref_vectors,
            )

            accumulated_batches += 1
            is_last_accum_step = (((i + 1) % accumulation_steps == 0)
                                  or (i + 1 == len(train_loader)))
            ctx = model.no_sync() if (args.distributed and not is_last_accum_step) else nullcontext()
            with ctx:
                loss.backward()
            accumulated_loss += loss.item()

            if is_last_accum_step:
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
                
                if is_rank0 and global_optimizer_step % 10 == 0:
                    with open(f"{model_output_path}/gradient_stats.csv", "a") as f:
                        if global_optimizer_step == 0 and not args.resume_weights:
                            f.write("epoch,step," + ",".join([f"{name}_grad_norm,{name}_mean,{name}_update_ratio"
                                                            for name in grad_stats.keys()]) + "\n")
                        row = f"{epoch},{global_optimizer_step},"
                        row += ",".join([f"{s['norm']},{s['mean']},{s['update_ratio']}"
                                        for s in grad_stats.values()])
                        f.write(row + "\n")

                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= accumulated_batches
                
                unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if args.use_cosine_lr and args.lr_step_unit == "optimizer_step":
                    scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(raw_model)

                optimizer_loss = accumulated_loss / accumulated_batches
                epoch_loss += optimizer_loss
                epoch_optimizer_steps += 1
                global_optimizer_step += 1
                if is_rank0:
                    optimizer_log = f"{model_output_path}/optimizer_steps.csv"
                    write_header = not os.path.exists(optimizer_log)
                    with open(optimizer_log, "a") as f:
                        if write_header:
                            f.write(
                                "optimizer_step,epoch,minibatch,loss,"
                                "unclipped_grad_norm,gradients_finite,learning_rate\n"
                            )
                        gradients_finite = bool(torch.isfinite(unclipped_grad_norm).item())
                        f.write(
                            f"{global_optimizer_step},{epoch},{i},{optimizer_loss},"
                            f"{float(unclipped_grad_norm)},{int(gradients_finite)},"
                            f"{optimizer.param_groups[0]['lr']}\n"
                        )
                accumulated_loss = 0
                accumulated_batches = 0
            
                if is_rank0 and global_optimizer_step % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Optimizer [{global_optimizer_step}], "
                        f"Loss: {(epoch_loss / epoch_optimizer_steps):.4f}"
                    )

                if optimizer_limit_reached(global_optimizer_step, args.max_optimizer_steps):
                    reached_optimizer_limit = True

                run_stability_probe(global_optimizer_step, epoch, i)
                
            del x_1, x_0, t, loss
            if reached_optimizer_limit:
                break

        epoch_completed = (i + 1 == len(train_loader))
        next_resume_epoch = epoch + 1 if epoch_completed else epoch
        next_resume_minibatch = 0 if epoch_completed else i + 1

        if args.distributed:
            loss_tensor = torch.tensor(epoch_loss / epoch_optimizer_steps, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            epoch_mean_loss = loss_tensor.item()
        else:
            epoch_mean_loss = epoch_loss / epoch_optimizer_steps
        losses.append(epoch_mean_loss)
        last_completed_epoch = epoch

        # Advance epoch-stepped schedules before serializing so the checkpoint is the exact
        # state observed at the start of the next epoch.
        if epoch_completed and args.use_cosine_lr and args.lr_step_unit == "epoch":
            scheduler.step()

        # Overwrite latest checkpoint so training can be resumed at any point.
        if is_rank0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "global_optimizer_step": global_optimizer_step,
                "config": run_config,
                "full_config": full_config,
                "rng_state": capture_rng_state(),
                "schedule_definition": schedule_definition,
                "resume_epoch": next_resume_epoch,
                "resume_minibatch": next_resume_minibatch,
            }
            if args.use_cosine_lr:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if ema is not None:
                ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt, f"{model_output_path}/models/latest_checkpoint.pth")

        if reached_optimizer_limit:
            break

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
            "epoch": last_completed_epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "global_optimizer_step": global_optimizer_step,
            "config": run_config,
            "full_config": full_config,
            "rng_state": capture_rng_state(),
            "schedule_definition": schedule_definition,
            "resume_epoch": next_resume_epoch,
            "resume_minibatch": next_resume_minibatch,
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

        jet_attr_model_loaded = (
            probe_jet_attr_model if probe_jet_attr_model is not None
            else jet_attributes.load_model(model_path=get_model_pth_path(args.output_path)).to(device)
        )

        # Vector-field visualization is opt-in (cfg.inference.vf_mode). Default "none"
        # since the frames only show E vs p_x (2 of 4 features) and are rarely opened.
        run_cfg_vf   = args.vf_mode in ("cfg", "both")
        run_nocfg_vf = args.vf_mode in ("nocfg", "both")
        if run_cfg_vf or run_nocfg_vf:
            try:
                vf_n_viz = args.n_viz_samples if args.num_particles < MAX_N_PARTICLES else 100
                if run_cfg_vf:
                    make_clear_folder(f"{model_output_path}/vf_viz_cfg")
                    generate_model_vector_field(
                        out_dir=f"{model_output_path}/vf_viz_cfg",
                        final_model=raw_model,
                        jet_attr_model=jet_attr_model_loaded,
                        X_test=X_test,
                        scale=final_scale,
                        n_jet_types=len(args.jet_types),
                        n_particles_per_jet=args.num_particles,
                        n_features_per_particle=NUM_PARTICLE_FEATURES,
                        n_viz_samples=vf_n_viz,
                        integration_steps=args.integration_steps,
                        use_cfg=True,
                        cfg_guidance_weight=2.0,
                        use_hyperbolic=args.use_hyperbolic,
                        use_reference_vectors=args.use_reference_vectors,
                    )
                if run_nocfg_vf:
                    make_clear_folder(f"{model_output_path}/vf_viz_nocfg")
                    generate_model_vector_field(
                        out_dir=f"{model_output_path}/vf_viz_nocfg",
                        final_model=raw_model,
                        jet_attr_model=jet_attr_model_loaded,
                        X_test=X_test,
                        scale=final_scale,
                        n_jet_types=len(args.jet_types),
                        n_particles_per_jet=args.num_particles,
                        n_features_per_particle=NUM_PARTICLE_FEATURES,
                        n_viz_samples=vf_n_viz,
                        integration_steps=args.integration_steps,
                        use_cfg=False,
                        use_hyperbolic=args.use_hyperbolic,
                        use_reference_vectors=args.use_reference_vectors,
                    )
            except Exception as e:
                print(f"Error occurred while generating model vector field: {e}")
                with open(f"{model_output_path}/error_log.txt", "a") as f:
                    f.write(f"Error occurred while generating model vector field: {e}\n")

        gen_jet_types = None
        gen_pt_cond = None
        try:
            random.seed(args.inference_seed)
            np.random.seed(args.inference_seed)
            torch.manual_seed(args.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.inference_seed)
            samples, gen_jet_types, gen_pt_cond, prior_samples = generate_samples(
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
                **generation_controls_from_namespace(args),
            )
        except Exception as e:
            print(f"Error occurred while generating samples: {e}")
            with open(f"{model_output_path}/error_log.txt", "a") as f:
                f.write(f"Error occurred while generating samples: {e}\n")
            exit(1)

        # Persist a small sample subset so plots/metrics can be regenerated without re-running.
        try:
            torch.save(samples[:10000].cpu(), f"{model_output_path}/samples_subset.pt")
            torch.save(prior_samples[:10000].cpu(), f"{model_output_path}/prior_samples_subset.pt")
        except Exception as e:
            print(f"Error saving sample subset: {e}")

        generation_diagnostics = None
        invalid_fraction = None
        if (args.max_invalid_fraction is not None
                or args.warn_invalid_fraction is not None
                or args.qualification_min_loss_improvement is not None):
            diagnostics_path = f"{model_output_path}/generation_diagnostics.json"
            if not os.path.isfile(diagnostics_path):
                raise RuntimeError(
                    "inference.max_invalid_fraction requires generation diagnostics"
                )
            with open(diagnostics_path) as handle:
                generation_diagnostics = json.load(handle)
            n_total = int(generation_diagnostics["n_total"])
            n_unstable = int(generation_diagnostics.get(
                "n_unstable", generation_diagnostics["n_failed"]
            ))
            invalid_fraction = n_unstable / max(n_total, 1)

        qualification_errors = []
        qualification_warnings = []
        qualification_summary = None
        if args.max_optimizer_steps is not None:
            optimizer_log = f"{model_output_path}/optimizer_steps.csv"
            optimizer_frame = pd.read_csv(optimizer_log)
            loss_summary = loss_improvement_summary(optimizer_frame["loss"].to_numpy())
            window = loss_summary["loss_window"]
            first_median = loss_summary["first_loss_median"]
            final_median = loss_summary["final_loss_median"]
            loss_improvement = loss_summary["loss_improvement_fraction"]
            losses_finite = loss_summary["losses_finite"]
            gradients_finite = bool(optimizer_frame["gradients_finite"].astype(bool).all())
            completed_required_steps = global_optimizer_step == args.max_optimizer_steps
            no_explosions = bool(
                generation_diagnostics is not None
                and int(generation_diagnostics["n_crossed_max_abs_1e6"]) == 0
            )
            qualification_summary = {
                "velocity_readout_init": args.velocity_readout_init,
                "global_optimizer_step": global_optimizer_step,
                "required_optimizer_steps": args.max_optimizer_steps,
                "completed_required_steps": completed_required_steps,
                "loss_window": window,
                "first_loss_median": first_median,
                "final_loss_median": final_median,
                "loss_improvement_fraction": loss_improvement,
                "minimum_loss_improvement_fraction": args.qualification_min_loss_improvement,
                "losses_finite": losses_finite,
                "gradients_finite": gradients_finite,
                "invalid_fraction": invalid_fraction,
                "warn_invalid_fraction": args.warn_invalid_fraction,
                "max_invalid_fraction": args.max_invalid_fraction,
                "no_explosions": no_explosions,
            }
            if not completed_required_steps:
                qualification_errors.append("did not reach required optimizer-step budget")
            if not losses_finite:
                qualification_errors.append("optimizer loss contains non-finite values")
            if not gradients_finite:
                qualification_errors.append("gradient norm contains non-finite values")
            if (args.qualification_min_loss_improvement is not None
                    and loss_improvement < args.qualification_min_loss_improvement):
                qualification_errors.append(
                    f"loss improvement {loss_improvement:.6f} is below "
                    f"{args.qualification_min_loss_improvement:.6f}"
                )
            if not no_explosions:
                qualification_errors.append("one or more trajectories crossed |x| > 1e6")

        if (args.max_invalid_fraction is not None and invalid_fraction is not None
                and invalid_fraction > args.max_invalid_fraction):
            qualification_errors.append(
                f"invalid_fraction={invalid_fraction:.6f} exceeds "
                f"max_invalid_fraction={args.max_invalid_fraction:.6f}"
            )
        if (args.warn_invalid_fraction is not None and invalid_fraction is not None
                and invalid_fraction > args.warn_invalid_fraction):
            warning = (
                f"invalid_fraction={invalid_fraction:.6f} exceeds warning threshold "
                f"{args.warn_invalid_fraction:.6f}"
            )
            qualification_warnings.append(warning)
            print(f"WARNING: {warning}; continuing to physics metrics")
        if qualification_summary is not None:
            qualification_summary["passed"] = not qualification_errors
            qualification_summary["errors"] = qualification_errors
            qualification_summary["warnings"] = qualification_warnings
            with open(f"{model_output_path}/qualification_summary.json", "w") as handle:
                json.dump(qualification_summary, handle, indent=2)
        if qualification_errors:
            raise RuntimeError("qualification gate failed: " + "; ".join(qualification_errors))

        eval_info = {}
        if not args.skip_metrics:
            try:
                eval_info = run_save_metrics(
                    X_test=X_test,
                    jet_types=args.jet_types,
                    gen_samples=samples,
                    output_path=model_output_path,
                    device=device,
                    gen_jet_types=gen_jet_types,
                    gen_pt_cond=gen_pt_cond,
                    prior_samples=prior_samples,
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
                "num_epochs": len(losses),
                "global_optimizer_step": global_optimizer_step,
                "qualification": qualification_summary,
                "git_commit": git_commit,
                # Provenance/scale knobs useful when comparing runs at a glance.
                "n_parameters": sum(p.numel() for p in raw_model.parameters()),
                "train_seconds": round(time.time() - train_start_time, 1),
                "hyperbolic_model": args.hyperbolic_model if args.use_hyperbolic else None,
                "regulator_mass": args.regulator_mass if (args.use_hyperbolic and args.hyperbolic_model == "mass_shell") else None,
                "effective_prior_dist": args.prior_dist,
                "effective_icp": {
                    "enabled": bool(args.use_icp),
                    "cache_path": icp_cache_path,
                },
                "generation": {
                    "seed": args.inference_seed,
                    "use_cfg": bool(args.use_cfg),
                    "cfg_guidance_weight": args.cfg_guidance_weight,
                    "integration_steps": args.integration_steps,
                },
                "config": run_config,
                "full_config": full_config,
                "metrics": {k: eval_info.get(k) for k in (
                    "w1m", "w1p", "w1efp", "fpd", "cov_mmd",
                    "frac_negative_energy", "frac_spacelike", "msq_median",
                    "isotropy_ks_costheta", "isotropy_ks_costheta_p",
                    "isotropy_ks_phi", "isotropy_ks_phi_p",
                    # FPND is stored per jet type under fpnd_{jet_type} (e.g. fpnd_g).
                    *(f"fpnd_{jt}" for jt in args.jet_types),
                )},
            }

            def _json_default(o):
                # numpy arrays (e.g. w1p, w1efp) aren't JSON-native and float() raises on them.
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if hasattr(o, "item"):        # numpy/torch scalar
                    return o.item()
                if hasattr(o, "__float__"):
                    return float(o)
                return str(o)

            with open(f"{model_output_path}/summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=_json_default)
        except Exception as e:
            print(f"Error writing summary.json: {e}")
