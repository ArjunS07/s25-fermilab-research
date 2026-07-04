import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from jetnet.utils import EtaPhiPtE_to_cartesian

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.file_management import make_clear_folder
from util.distributions import gen_initial_distribution
from util.hyperbolic import to_poincare_ball, from_poincare_ball

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]


def build_reference_vectors(jet_eta, jet_pt, final_scale, device):
    """Inference reference 4-vectors: e_t=(1,0,0,0) and a reconstructed jet 4-momentum.

    Mirrors util/coordinates.transform_rel_particle_coordinates_to_cartesian: the jet axis is
    (jet_eta, random phi, jet_pt) with energy E = pt*cosh(eta), converted with the same jetnet
    routine used to build the training particles, then divided by final_scale to enter the
    model's scaled space. This matches train.py, where the reference is the sum of the scaled
    constituents (= physical jet 4-momentum / final_scale). The random phi is the *chosen* jet
    orientation the references then induce in the generated cloud.

    Returns (B, 2, 4).
    """
    batch = jet_eta.shape[0]
    phi = (2 * torch.pi) * torch.rand(batch, device=device)
    energy = jet_pt * torch.cosh(jet_eta)
    stacked = torch.stack([jet_eta, phi, jet_pt, energy], dim=-1)  # (B, 4) = (eta, phi, pt, E)
    jet_p4 = EtaPhiPtE_to_cartesian(stacked) / final_scale         # (B, 4) scaled (E, px, py, pz)
    e_t = torch.zeros(batch, 4, device=device, dtype=jet_p4.dtype)
    e_t[:, 0] = 1.0
    return torch.stack([e_t, jet_p4], dim=1)  # (B, 2, 4)

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
):


    # make folder
    make_clear_folder(f"{root_output_path}/samples")

    all_samples = []
    all_pt_cond = []   # conditioning pT for each jet (normalized)
    all_masks = []     # particle masks for each jet

    with torch.no_grad():
        model.eval()
        jet_attr_model.eval()

        times = torch.linspace(0, 1, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)
            x = gen_initial_distribution(
                batch_size=current_batch_size,
                num_particles=max_particles_per_jet,
                device=device
            )
            x = x.to(device)

            generated_jet_attrs, _ = jet_attributes.generate_jets(
                jet_attr_model, device, n_jet_types=n_jet_types, num_jets=x.shape[0]
            )
            # generated_jet_attrs layout: [one_hot(5), eta, pt, mass, n_particles]
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            gen_pt = generated_jet_attrs[:, 6].to(device)   # normalized pT from NF
            gen_n_particles = generated_jet_attrs[:, -1].long().to(device)
            gen_n_particles = gen_n_particles.clamp(max=max_particles_per_jet)

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

            scaled_x = final_scale * x * masks.unsqueeze(-1)   # zero-out padded slots
            # torch.save(scaled_x, f"{root_output_path}/samples/batch_{start_idx//batch_size:04d}.pt")

            all_samples.append(scaled_x.cpu())
            all_pt_cond.append(gen_pt.cpu())
            all_masks.append(masks.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_samples_cat = torch.cat(all_samples, dim=0)
    all_pt_cond_cat = torch.cat(all_pt_cond, dim=0)
    all_masks_cat = torch.cat(all_masks, dim=0)

    _plot_pt_comparison(
        all_samples_cat, all_pt_cond_cat, all_masks_cat,
        out_path=f"{root_output_path}/samples/pt_comparison.png"
    )

    return all_samples_cat.to(device)


def _plot_pt_comparison(samples, pt_cond, masks, out_path):
    """
    Scatter plot of sum-of-particle-pT vs conditioning pT per jet.

    samples : (N, max_particles, 4)  Cartesian (E, px, py, pz), physical units
    pt_cond : (N,)                   Normalized conditioning pT from the NF
    masks   : (N, max_particles)     1 for real particles, 0 for padding
    """
    # Per-particle pT = sqrt(px^2 + py^2), then sum over real particles
    px = samples[:, :, 1]
    py = samples[:, :, 2]
    particle_pt = torch.sqrt(px ** 2 + py ** 2)
    sum_pt = (particle_pt * masks).sum(dim=1).numpy()
    pt_cond_np = pt_cond.numpy()

    slope, intercept, r_value, _, _ = stats.linregress(pt_cond_np, sum_pt)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        x=pt_cond_np, y=sum_pt,
        alpha=0.3, s=8, linewidth=0,
        color=sns.color_palette("deep")[0],
        ax=ax, label="Generated jets"
    )
    x_line = np.linspace(pt_cond_np.min(), pt_cond_np.max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color="crimson", linewidth=1.5,
            label=rf"Linear fit ($R^2={r_value**2:.3f}$, slope=${slope:.3f}$)")
    ax.set_xlabel(r"Conditioning $p_T$ (normalized)", fontsize=13)
    ax.set_ylabel(r"$\sum_i p_{T,i}$ (generated, physical units)", fontsize=13)
    ax.set_title(r"Generated scalar $p_T$ sum vs. conditioning $p_T$", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"pT comparison plot saved to {out_path}")


