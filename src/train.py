import os
import json
import time
import pickle
import logging
import math

import numpy as np
import torch
import random
from torch.utils.data import DataLoader

from models.lorentznet_flow import build_lorentznet
from util.data.jet_attributes import NUM_CLASSES
from util.data.distributions import gen_initial_distribution, time_dist
from util.geometry.coordinates import (deterministic_jet_phi,
                              transform_rel_particle_coordinates_to_cartesian)
from util.geometry.conditioning import scale_condition_pt
from util.infra.ema import ModelEMA
from util.infra.file_management import make_clear_folder
from data import get_data_path
from util.geometry.coordinates import build_reference_vectors
from util.geometry.online_coupling import online_geodesic_coupling
from util.metrics.qualification import optimizer_limit_reached
from util.infra.rng import capture_rng_state, keyed_seed, keyed_torch_rng, restore_rng_state
from util.infra.checkpoint_config import build_run_config, build_checkpoint
from util.infra.grad_stats import collect_gradient_stats
from util.infra.lr_schedule import build_step_scheduler
from training import flow_matching_loss
from training.stability_probe import run_stability_probe
from config import TrainRunConfig, build_config, parse_config_cli
from training.final_reporting import finalize_training

RANDOM_SEED = 42
MAX_N_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, jet_info, particle_data, jet_phi):
        self.jet_info = jet_info
        self.particle_data = particle_data
        self.jet_phi = jet_phi

    def __len__(self):
        return len(self.particle_data)

    def __getitem__(self, idx):
        return self.jet_info[idx], self.particle_data[idx], self.jet_phi[idx]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)    
    if "TORCH_NUM_THREADS" in os.environ:
        torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
    if "TORCH_NUM_INTEROP_THREADS" in os.environ:
        torch.set_num_interop_threads(int(os.environ["TORCH_NUM_INTEROP_THREADS"]))
    print(
        f"CPU thread pools: intra_op={torch.get_num_threads()} "
        f"inter_op={torch.get_num_interop_threads()}"
    )
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    config_path, overrides = parse_config_cli()
    cfg = build_config(TrainRunConfig, config_path, overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    data_path = get_data_path(cfg.paths.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(f"{data_path}/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    model_output_path = f"{cfg.paths.output_path}/train"
    if cfg.paths.resume_weights:
        os.makedirs(model_output_path, exist_ok=True)
    else:
        make_clear_folder(model_output_path)
        with open(f"{model_output_path}/config.txt", "w") as f:
            json.dump(cfg.model_dump(), f, indent=2, default=str)

    train_jet_phi = deterministic_jet_phi(len(X_train), RANDOM_SEED)
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(
        X_train, jet_phi=train_jet_phi).to('cpu')
    X_train_particle_transformed = X_train_particle_transformed[:cfg.training.n_train_samples]
    if cfg.data.num_particles < MAX_N_PARTICLES:
        # Particles are, by default, ordered by p_t. take the n highest pt particles in each jet
        X_train_particle_transformed = X_train_particle_transformed[:, :cfg.data.num_particles, :]
    
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
    # Model initialization is an isolated treatment stream. Its parameter count can no
    # longer perturb data order, time, dropout, or prior draws in another arm.
    torch.manual_seed(cfg.training.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.model_seed)
    model = build_lorentznet(
        NUM_CLASSES,
        num_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.n_hidden,
        regulator_mass=cfg.model.regulator_mass,
    ).to(device)
    
    start_epoch = 0
    resume_minibatch = 0
    losses = []
    global_optimizer_step = 0
    if cfg.paths.resume_weights:
        # Resume checkpoints are trusted first-party artifacts and include full
        # optimizer, scheduler, config, and Python/NumPy/Torch RNG state.
        checkpoint = torch.load(cfg.paths.resume_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("resume_epoch", checkpoint["epoch"] + 1))
        resume_minibatch = int(checkpoint.get("resume_minibatch", 0))
        losses = checkpoint.get("losses", [])
        global_optimizer_step = int(checkpoint.get("global_optimizer_step", 0))
        print(f"Resumed from checkpoint at epoch {start_epoch - 1}; "
              f"running {cfg.training.num_epochs} more epochs ({start_epoch}→{start_epoch + cfg.training.num_epochs - 1}).")

    # EMA of weights (Phase 2.1). Shadow starts from the (possibly resumed) weights.
    ema = None
    if cfg.training.use_ema:
        ema = ModelEMA(model, decay=cfg.training.ema_decay)
        if cfg.paths.resume_weights and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"], device=device)
            print("Resumed EMA shadow weights from checkpoint.")

    # Self-describing model config, embedded in every checkpoint so a run can be resumed or
    # loaded for inference without re-specifying the architecture flags on the CLI.
    run_config = build_run_config(cfg, final_scale)
    # Full config (all sections), embedded alongside the architecture-only `config` dict
    # so infer.py can auto-load every knob a run was trained with.
    full_config = cfg.model_dump()
    if cfg.paths.resume_weights:
        prev = checkpoint.get("config")
        if prev is not None:
            mism = {k: (prev.get(k), run_config.get(k))
                    for k in ("n_layers", "n_hidden", "num_particles", "regulator_mass")
                    if prev.get(k) != run_config.get(k)}
            if mism:
                print(f"WARNING: resume architecture flags differ from checkpoint: {mism}. "
                      f"Re-run with matching flags or the loaded weights are wrong.")

    if not cfg.paths.resume_weights:
        make_clear_folder(f"{model_output_path}/models")
    else:
        os.makedirs(f"{model_output_path}/models", exist_ok=True)

    train_jet_info = X_train[:][1].to(device)
    if cfg.data.num_particles < MAX_N_PARTICLES:
        train_jet_info[:, 3] = train_jet_info[:, 3].clamp(max=cfg.data.num_particles)
    train_jet_info = train_jet_info[:cfg.training.n_train_samples]

    # ── Coupling ────────────────────────────────────────────────────────────────
    # The frozen ICP cache has been removed (a fixed per-jet prior/pairing reused every epoch
    # let the field specialise to a finite bundle of paths; see discussions/22). Training draws
    # fresh prior noise per step and applies the geodesic ICP coupling *online*
    # (util.geometry.online_coupling), so the supervision measure is the fresh-noise marginal field.
    print(f"Coupling: {cfg.training.coupling} (fresh noise every step; no frozen cache)")

   
    def _sample_t(batch_size: int) -> torch.Tensor:
        mode = cfg.training.time_sampling if cfg.training.use_time_sampling else 'uniform'
        if mode == 'uniform':
            return time_dist(batch_size, device=device, mode='uniform')
        elif mode == 'power_law':
            return time_dist(batch_size, device=device, mode='power_law', a=-0.2)
        elif mode == 'lognorm':
            return time_dist(batch_size, device=device, mode='lognorm', mu=-0.5, sigma=1.0)
        else:
            raise ValueError(f"Unknown time_sampling: {mode}")

    epoch_fraction = cfg.training.epoch_frac
    samples_per_epoch = int(epoch_fraction * len(X_train_particle_transformed))
    accumulation_steps = cfg.training.target_batch_size // cfg.training.batch_size
    minibatches_per_epoch = math.ceil(samples_per_epoch / cfg.training.batch_size)
    optimizer_steps_per_epoch = math.ceil(minibatches_per_epoch / accumulation_steps)

    lr = cfg.training.lr
    weight_decay = cfg.training.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    planned_optimizer_steps = (
        cfg.training.max_optimizer_steps
        if cfg.training.max_optimizer_steps is not None
        else cfg.training.num_epochs * optimizer_steps_per_epoch
    )
    schedule_definition = {
        "planned_optimizer_steps": planned_optimizer_steps,
        "warmup_steps": cfg.training.lr_warmup_steps,
        "eta_min_factor": cfg.training.eta_min_factor,
    }
    if cfg.paths.resume_weights and checkpoint.get("schedule_definition") is not None:
        previous_schedule = checkpoint["schedule_definition"]
        if previous_schedule != schedule_definition:
            raise ValueError(
                "resume schedule definition differs from checkpoint: "
                f"checkpoint={previous_schedule}, requested={schedule_definition}"
            )

    if cfg.training.use_cosine_lr:
        scheduler = build_step_scheduler(
            optimizer,
            total_steps=planned_optimizer_steps,
            warmup_steps=cfg.training.lr_warmup_steps,
            eta_min_factor=cfg.training.eta_min_factor,
        )

    if cfg.paths.resume_weights:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if cfg.training.use_cosine_lr and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))

    probe_steps = set(cfg.training.stability_probe_steps)
    probe_jet_attr_model = None

    probe_jet_attr_model = run_stability_probe(
        optimizer_step=global_optimizer_step, epoch=start_epoch - 1, minibatch=None,
        probe_steps=probe_steps, model=model, optimizer=optimizer, scheduler=scheduler,
        ema=ema, losses=losses, cfg=cfg, device=device, model_output_path=model_output_path,
        train_loader=None, run_config=run_config, full_config=full_config,
        schedule_definition=schedule_definition, final_scale=final_scale,
        jet_attr_model=probe_jet_attr_model,
    )

    
    total_epochs = start_epoch + cfg.training.num_epochs
    train_start_time = time.time()
    reached_optimizer_limit = optimizer_limit_reached(
        global_optimizer_step, cfg.training.max_optimizer_steps
    )
    last_completed_epoch = start_epoch - 1
    next_resume_epoch = start_epoch
    next_resume_minibatch = resume_minibatch
    for epoch in range(start_epoch, total_epochs):
        if reached_optimizer_limit:
            break
        epoch_loss = 0
        epoch_optimizer_steps = 0

        # ── Sample epoch indices ──────────────────────────────────────────────
        # Deterministically draw a fresh uniform subset for this epoch.
        selection_generator = torch.Generator(device="cpu")
        selection_generator.manual_seed(keyed_seed(cfg.training.data_order_seed, epoch))

        epoch_indices = torch.randperm(
            len(X_train_particle_transformed), generator=selection_generator,
        )[:samples_per_epoch]

        X_train_epoch = torch.utils.data.Subset(X_train_particle_transformed, epoch_indices)
        train_jet_info_epoch = train_jet_info[epoch_indices]
        train_jet_phi_epoch = train_jet_phi[epoch_indices]

        paired_dataset = PairedDataset(
            train_jet_info_epoch, X_train_epoch, train_jet_phi_epoch
        )
        train_loader = DataLoader(
            paired_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            pin_memory=False,
            generator=torch.Generator(device="cpu").manual_seed(
                keyed_seed(cfg.training.data_order_seed, epoch, 1)
            ),
        )

        optimizer.zero_grad()
        accumulation_steps = cfg.training.target_batch_size // cfg.training.batch_size
        accumulated_losses = []
        accumulated_batches = 0
        accumulated_field = {
            "n_real": 0, "target_norm_sum": 0.0, "target_norm_sq_sum": 0.0,
            "prediction_norm_sum": 0.0, "prediction_norm_sq_sum": 0.0,
        }

        for i, (batch_jet_info, batch_particle_info, batch_jet_phi) in enumerate(train_loader):
            if epoch == start_epoch and i < resume_minibatch:
                continue

            batch_jet_info = batch_jet_info.to(device)
            batch_particle_info = batch_particle_info.to(device)
            batch_jet_phi = batch_jet_phi.to(device)

            batch_jet_n_particles = batch_jet_info[:, 3]
            batch_jet_pt = batch_jet_info[:, 1]  # normalized pT from FeaturewiseLinear
            batch_jet_mass = batch_jet_info[:, 2]
            encoded_jet_types = jet_attributes.one_hot_enc_jet_type(batch_jet_info[:, 4].long())
            # Legacy checkpoints consume raw pT. v2 uses model-scaled pT and mass.
            cond_pt = scale_condition_pt(batch_jet_pt, final_scale)
            condition_parts = [
                encoded_jet_types,
                batch_jet_n_particles.unsqueeze(-1),
                cond_pt.unsqueeze(-1),
            ]
            condition_parts.append((batch_jet_mass / final_scale).unsqueeze(-1))
            batch_jet_info_cropped = torch.cat(condition_parts, dim=-1)

            # Per-sample CFG dropout: each sample independently drops jet type + pT
            # conditioning. n_particles is preserved because it is already encoded in
            # the particle mask, so nulling it adds no guidance signal and only weakens
            # the unconditional-conditional gap.
            dropout_rate = cfg.training.cfg_null_dropout_rate
            if dropout_rate <= 0:
                dropout_mask = None
            elif dropout_rate >= 1:
                dropout_mask = torch.ones(
                    batch_jet_info_cropped.shape[0], dtype=torch.bool, device=device
                )
            else:
                with keyed_torch_rng(cfg.training.dropout_seed, epoch, i, device):
                    dropout_mask = (
                        torch.rand(batch_jet_info_cropped.shape[0], device=device)
                        < dropout_rate
                    )
            if dropout_mask is not None and (dropout_rate >= 1 or dropout_mask.any()):
                null_for_batch = model.make_null_cond(batch_jet_info_cropped)
                batch_jet_info_cropped = torch.where(
                    dropout_mask.unsqueeze(-1), null_for_batch, batch_jet_info_cropped
                )

            x_1 = batch_particle_info[:, :, :4]
            true_masks = batch_particle_info[:, :, 4]   # always use mask
            # Fresh prior noise every step (no frozen cache → no path memorization).
            with keyed_torch_rng(cfg.training.prior_seed, epoch, i, device):
                x_0 = gen_initial_distribution(
                    x_1=x_1, prior_dist=cfg.training.prior_dist,
                    jet_features=batch_jet_info, jet_phi=batch_jet_phi,
                    device=device, model_scale=final_scale, particle_mask=true_masks,
                )
            # Online geodesic ICP coupling on the freshly drawn noise (minibatch OT-CFM).
            if cfg.training.coupling == "online_geodesic_icp":
                x_0 = online_geodesic_coupling(x_0, x_1, true_masks, cfg.model.regulator_mass)

            mask_exp = true_masks.unsqueeze(-1).expand(-1, -1, NUM_PARTICLE_FEATURES)
            x_1 = mask_exp * x_1
            x_0 = mask_exp * x_0

            # External physical massless conditioning jet.  This is reproducible from the
            # same attributes at inference and never leaks the target constituent sum.
            ref_vectors = build_reference_vectors(
                batch_jet_info[:, 0], batch_jet_info[:, 1], final_scale, device,
                jet_phi=batch_jet_phi, jet_mass=batch_jet_info[:, 2],
            )

            with keyed_torch_rng(cfg.training.time_seed, epoch, i, device):
                t = _sample_t(x_0.shape[0])
            loss_result = flow_matching_loss(
                model=model, raw_model=model, config=cfg.model,
                x0=x_0, x1=x_1, t=t, mask=true_masks,
                conditions=batch_jet_info_cropped, references=ref_vectors,
            )
            if isinstance(loss_result, tuple):
                loss, field_stats = loss_result
                for key in accumulated_field:
                    accumulated_field[key] += field_stats[key]
            else:
                loss = loss_result

            accumulated_batches += 1
            is_last_accum_step = (((i + 1) % accumulation_steps == 0)
                                  or (i + 1 == len(train_loader)))
            loss.backward()
            accumulated_losses.append(loss.detach())

            if is_last_accum_step:
                if global_optimizer_step % 10 == 0:
                    grad_stats = collect_gradient_stats(model)
                    with open(f"{model_output_path}/gradient_stats.csv", "a") as f:
                        if global_optimizer_step == 0 and not cfg.paths.resume_weights:
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
                if cfg.training.use_cosine_lr:
                    scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

                # Transfer all accumulated scalar losses together, then preserve the
                # historical ordered Python-float summation exactly.
                accumulated_loss = sum(torch.stack(accumulated_losses).cpu().tolist())
                optimizer_loss = accumulated_loss / accumulated_batches
                epoch_loss += optimizer_loss
                epoch_optimizer_steps += 1
                global_optimizer_step += 1
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
                if accumulated_field["n_real"]:
                    n_field = accumulated_field["n_real"]
                    target_mean = accumulated_field["target_norm_sum"] / n_field
                    prediction_mean = accumulated_field["prediction_norm_sum"] / n_field
                    target_var = max(
                        accumulated_field["target_norm_sq_sum"] / n_field
                        - target_mean * target_mean, 0.0
                    )
                    prediction_var = max(
                        accumulated_field["prediction_norm_sq_sum"] / n_field
                        - prediction_mean * prediction_mean, 0.0
                    )
                    field_log = f"{model_output_path}/conditional_field_stats.csv"
                    write_field_header = not os.path.exists(field_log)
                    with open(field_log, "a") as f:
                        if write_field_header:
                            f.write(
                                "optimizer_step,epoch,minibatch,n_real,"
                                "target_norm_mean,target_norm_variance,"
                                "prediction_norm_mean,prediction_norm_variance\n"
                            )
                        f.write(
                            f"{global_optimizer_step},{epoch},{i},{n_field},"
                            f"{target_mean},{target_var},{prediction_mean},"
                            f"{prediction_var}\n"
                        )
                accumulated_losses.clear()
                accumulated_batches = 0
                for key in accumulated_field:
                    accumulated_field[key] = 0
            
                if global_optimizer_step % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Optimizer [{global_optimizer_step}], "
                        f"Loss: {(epoch_loss / epoch_optimizer_steps):.4f}"
                    )

                if optimizer_limit_reached(global_optimizer_step, cfg.training.max_optimizer_steps):
                    reached_optimizer_limit = True

                probe_jet_attr_model = run_stability_probe(
                    optimizer_step=global_optimizer_step, epoch=epoch, minibatch=i,
                    probe_steps=probe_steps, model=model, optimizer=optimizer, scheduler=scheduler,
                    ema=ema, losses=losses, cfg=cfg, device=device,
                    model_output_path=model_output_path, train_loader=train_loader,
                    run_config=run_config, full_config=full_config,
                    schedule_definition=schedule_definition, final_scale=final_scale,
                    jet_attr_model=probe_jet_attr_model,
                )
                
            del x_1, x_0, t, loss
            if reached_optimizer_limit:
                break

        epoch_completed = (i + 1 == len(train_loader))
        next_resume_epoch = epoch + 1 if epoch_completed else epoch
        next_resume_minibatch = 0 if epoch_completed else i + 1

        epoch_mean_loss = epoch_loss / epoch_optimizer_steps
        losses.append(epoch_mean_loss)
        last_completed_epoch = epoch

        # Overwrite latest checkpoint so training can be resumed at any point.
        ckpt = build_checkpoint(
            model_state=model.state_dict(), epoch=epoch,
            global_optimizer_step=global_optimizer_step, losses=losses,
            run_config=run_config, full_config=full_config,
            optimizer_state=optimizer.state_dict(), rng_state=capture_rng_state(),
            scheduler_state=scheduler.state_dict() if cfg.training.use_cosine_lr else None,
            ema_state=ema.state_dict() if ema is not None else None,
            extra={"schedule_definition": schedule_definition,
                   "resume_epoch": next_resume_epoch,
                   "resume_minibatch": next_resume_minibatch},
        )
        torch.save(ckpt, f"{model_output_path}/models/latest_checkpoint.pth")

        if reached_optimizer_limit:
            break

    training_seconds = round(time.time() - train_start_time, 1)
    # Logging — on resume, append only the newly-completed epochs.
    write_mode = "a" if cfg.paths.resume_weights else "w"
    with open(f"{model_output_path}/training_loss.csv", write_mode) as f:
        if not cfg.paths.resume_weights:
            f.write("epoch,loss\n")
        for epoch_i, loss_val in enumerate(losses[start_epoch:], start=start_epoch):
            f.write(f"{epoch_i},{loss_val}\n")

    finalize_training(
        cfg=cfg, model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
        losses=losses, last_completed_epoch=last_completed_epoch,
        global_optimizer_step=global_optimizer_step, run_config=run_config,
        full_config=full_config, schedule_definition=schedule_definition,
        next_resume_epoch=next_resume_epoch, next_resume_minibatch=next_resume_minibatch,
        model_output_path=model_output_path, final_scale=final_scale,
        training_seconds=training_seconds, x_test=X_test, device=device,
        max_n_particles=MAX_N_PARTICLES, jet_attr_model=probe_jet_attr_model,
    )
