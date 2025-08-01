import argparse
import torch
from util import jet_attributes
from models.NewLEFM import LEJetGeneratorFM
import matplotlib.pyplot as plt
import seaborn as sns

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]

def generate_samples(
        model,
        device,
        out_dir,
        num_particles,
        num_particle_features,
        final_scale,
        integration_steps,
        n_samples,
        batch_size,
        n_jet_types=3,
):

    jet_attr_generator = jet_attributes.load_model().to(device)
    with torch.no_grad():
        model.eval()
        jet_attr_generator.eval()
        
        times = torch.linspace(0, 1, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)
            x = torch.randn(
                (current_batch_size, num_particles, num_particle_features),
                device=device
            )
            generated_jet_attrs, _ = jet_attributes.generate_jets(jet_attr_generator, device, n_jet_types=n_jet_types, num_jets=x.shape[0])
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            n_particles = generated_jet_attrs[:, -1].long().to(device)

            masks = jet_attributes.generate_masks(
                n_particles,
                max_n_particles=num_particles,
                device=device
            )
            generated_jet_attrs = torch.cat([
                jet_one_hot_enc,
                n_particles.unsqueeze(-1).float()
            ], dim=-1)

            for i in range(integration_steps):
                new_x = model.step(x, generated_jet_attrs, masks, times[i], times[i + 1])
                update = new_x - x
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
                plt.savefig(f"{out_dir}/samples_histogram_{start_idx//batch_size:04d}_step_{i}.png")
                plt.close(fig)

            torch.save(final_scale * x, f"{out_dir}/samples_batch_{start_idx//batch_size:04d}.pt")

            if start_idx % (batch_size * 10) == 0:
                print(f"Generated {start_idx + batch_size} samples so far")
            
            del x, generated_jet_attrs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples using the trained model")
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")

    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save generated samples")
    parser.add_argument("--num_particles", type=int, default=150, help="Number of particles in each jet")
    parser.add_argument("--num_particle_features", type=int, default=4, help="Number of features per particle")
    parser.add_argument("--final_scale", type=float, default=42.0, help="Final scale factor for the generated samples")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generating samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")

    args = parser.parse_args()
    model_info = torch.load(args.model_path, map_location=args.device)
    n_layers = 1 + int(list(model_info.keys())[-71].split("layers.")[-1].split(".")[0])
    model = LEJetGeneratorFM(n_layers=n_layers).to(args.device)
    model.load_state_dict(model_info)  # Load the model state dictionary

    model.to(args.device)
    model.eval()

    generate_samples(
        model=model,
        device=args.device,
        out_dir=args.out_dir,
        num_particles=args.num_particles,
        num_particle_features=args.num_particle_features,
        final_scale=args.final_scale,
        integration_steps=args.integration_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size
    )
    print(f"Samples generated and saved to {args.out_dir}/gen/samples_cartesian.pt")