import seaborn as sns
import torch
import matplotlib.pyplot as plt

from jetnet.utils import EtaPhiPtE_to_cartesian

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.file_management import make_clear_folder
from util.distributions import gen_initial_distribution
from util.hyperbolic import to_poincare_ball, from_poincare_ball
from util.coordinates import build_reference_vectors

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]


def generate_samples(
        model: LEFTJeN,
        jet_attr_model,
        device,
        root_output_path,
        max_particles_per_jet,
        final_scale,
        integration_steps,
        n_samples,
        batch_size,
        n_jet_types=3,
        use_cfg=False,
        cfg_guidance_weight=2.0,
        use_hyperbolic=False,
        hyperbolic_c=1.0,
        hyperbolic_model='poincare',
        regulator_mass=0.5,
        use_reference_vectors=False,
        sampler='euler',
        prior_dist='isotropic_com',
):


    # make folder
    make_clear_folder(f"{root_output_path}/samples")

    all_samples = []
    all_pt_cond = []      # conditioning pT for each jet (normalized)
    all_jet_types = []    # per-jet class index (argmax of one-hot)

    with torch.no_grad():
        model.eval()
        jet_attr_model.eval()

        times = torch.linspace(0, 1 - 1e-5, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)

            generated_jet_attrs, _ = jet_attributes.generate_jets(
                jet_attr_model, device, n_jet_types=n_jet_types, num_jets=current_batch_size
            )
            # generated_jet_attrs layout: [one_hot(5), eta, pt, mass, n_particles]
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            gen_pt = generated_jet_attrs[:, 6].to(device)   # normalized pT from NF
            gen_n_particles = generated_jet_attrs[:, -1].long().to(device)
            gen_n_particles = gen_n_particles.clamp(max=max_particles_per_jet)

            # Build jet_features for priors that need jet-axis info.
            jet_features = None
            if prior_dist in ('axis_aligned', 'jet_ref_frame'):
                gen_eta = generated_jet_attrs[:, 5].to(device)
                gen_pt_prior = gen_pt
                jet_features = torch.stack([gen_eta, gen_pt_prior], dim=-1)

            x = gen_initial_distribution(
                batch_size=current_batch_size,
                num_particles=max_particles_per_jet,
                prior_dist=prior_dist,
                jet_features=jet_features,
                device=device
            )
            x = x.to(device)

            masks = jet_attributes.generate_masks(
                gen_n_particles,
                max_particles_per_jet=max_particles_per_jet,
                device=device
            )
            # Conditioning vector: [one_hot_type, n_particles, pT]
            cond = torch.cat([
                jet_one_hot_enc,
                gen_n_particles.unsqueeze(-1).float(),
                gen_pt.unsqueeze(-1),
            ], dim=-1).to(device)

            # Reference virtual particles, reconstructed from the sampled jet attributes.
            ref_vectors = None
            if use_reference_vectors:
                gen_eta = generated_jet_attrs[:, 5].to(device)  # jet eta (layout: [onehot(5), eta, pt, mass, n])
                ref_vectors = build_reference_vectors(gen_eta, gen_pt, final_scale, device)

            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                # Integrate the ODE geodesically on the mass shell. The state stays on H_m
                # (Cartesian on-shell 4-vectors), so the final x is used directly.
                from util.mass_shell import project_to_shell
                y = project_to_shell(x * masks.unsqueeze(-1), regulator_mass)
                for i in range(integration_steps):
                    y = model.step_hyperbolic(
                        y_t=y,
                        jet_conditions=cond,
                        mask=masks,
                        t_start=times[i],
                        t_end=times[i + 1],
                        hyperbolic_model='mass_shell',
                        regulator_mass=regulator_mass,
                        use_cfg=use_cfg,
                        guidance_weight=cfg_guidance_weight,
                        ref_vectors=ref_vectors,
                    )
                x = y
            elif use_hyperbolic:
                y = to_poincare_ball(x, c=hyperbolic_c)
                for i in range(integration_steps):
                    y = model.step_hyperbolic(
                        y_t=y,
                        jet_conditions=cond,
                        mask=masks,
                        t_start=times[i],
                        t_end=times[i + 1],
                        c=hyperbolic_c,
                        use_cfg=use_cfg,
                        guidance_weight=cfg_guidance_weight,
                        ref_vectors=ref_vectors,
                    )
                x = from_poincare_ball(y, c=hyperbolic_c)
            else:
                for i in range(integration_steps):
                    x = model.step(x, cond, masks, times[i], times[i + 1], method=sampler,
                                   use_cfg=use_cfg, guidance_weight=cfg_guidance_weight,
                                   ref_vectors=ref_vectors)

            x = x.clamp(-50, 50)
            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                # Component clamping is an inference guard, but a Cartesian clamp alone moves
                # points off H_m. Restore energy from the clamped spatial momentum.
                x = project_to_shell(x, regulator_mass)
            scaled_x = final_scale * x * masks.unsqueeze(-1)   # zero-out padded slots
            # torch.save(scaled_x, f"{root_output_path}/samples/batch_{start_idx//batch_size:04d}.pt")

            all_samples.append(scaled_x.cpu())
            all_pt_cond.append(gen_pt.cpu())
            all_jet_types.append(jet_one_hot_enc.argmax(dim=-1).cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_samples_cat   = torch.cat(all_samples, dim=0)
    all_pt_cond_cat   = torch.cat(all_pt_cond, dim=0)
    all_jet_types_cat = torch.cat(all_jet_types, dim=0)

    return all_samples_cat.to(device), all_jet_types_cat, all_pt_cond_cat
