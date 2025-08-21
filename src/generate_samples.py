import argparse
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from util import jet_attributes
from util.file_management import make_clear_folder
from util.distributions import gen_initial_distribution

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]

def generate_samples(
        model,
        jet_attr_model,
        device,
        root_output_path,
        num_particles,
        final_scale,
        integration_steps,
        n_samples,
        batch_size,
        n_jet_types=3,
        use_cfg=False,
        cfg_guidance_weight=2.0
):
    

    # make folder
    make_clear_folder(f"{root_output_path}/samples")

    all_samples = []

    with torch.no_grad():
        model.eval()
        jet_attr_model.eval()
        
        times = torch.linspace(0, 1, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)
            x = gen_initial_distribution(
                batch_size=current_batch_size,
                num_particles=num_particles,
                )
            x = x.to(device)
            
            generated_jet_attrs, _ = jet_attributes.generate_jets(jet_attr_model, device, n_jet_types=n_jet_types, num_jets=x.shape[0])
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            n_particles = generated_jet_attrs[:, -1].long().to(device)

            masks = jet_attributes.generate_masks(
                n_particles,
                max_particles_per_jet=num_particles,
                device=device
            )
            generated_jet_attrs = torch.cat([
                jet_one_hot_enc,
                n_particles.unsqueeze(-1).float()
            ], dim=-1)

            for i in range(integration_steps):
                new_x = model.step(x, generated_jet_attrs, masks, times[i], times[i + 1], use_cfg=use_cfg, guidance_weight=cfg_guidance_weight)
                x = new_x
                fig, axs = plt.subplots(1, 4, figsize=(20, 10))
                
                for j, feature in enumerate(features):
                    ax = axs[j]
                    sns.histplot(
                        x[:, :, j].flatten().cpu().numpy(),
                        bins=100,
                        ax=ax,
                        stat="density",
                        kde=True,
                        label="Generated"
                    )
                    ax.set_title(feature)
                plt.savefig(f"{root_output_path}/samples/histogram_{start_idx//batch_size:04d}_step_{i}.png")
                plt.close(fig)

            torch.save(final_scale * x, f"{root_output_path}/samples/batch_{start_idx//batch_size:04d}.pt")

            if start_idx % (batch_size * 10) == 0:
                print(f"Generated {start_idx + batch_size} samples so far")
            
            all_samples.append(final_scale * x.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return torch.cat(all_samples, dim=0)


