import argparse
import torch
from util import jet_attributes
from models.ConditionalLEFlowMatching import JetFMGenerator


def generate_samples(
        model,
        device,
        out_dir,
        num_particles,
        num_particle_features,
        final_scale,
        integration_steps,
        n_samples,
        batch_size
):
    samples = []
    jet_attr_generator = jet_attributes.load_model().to(device)
    with torch.no_grad():
        model.eval()
        jet_attr_generator.eval()
        
        times = torch.linspace(0, 1, integration_steps + 1).to(device)
        x = torch.randn((n_samples, num_particles, num_particle_features), device=device)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = x[start_idx:end_idx].to(device)
            generated_jet_attrs, _ = jet_attributes.generate_jets(jet_attr_generator, device, n_jet_types=1, num_jets=x_batch.shape[0])
            generated_jet_attrs = generated_jet_attrs.to(device)

            for i in range(integration_steps):
                x_batch = model.step(x_batch, generated_jet_attrs, times[i], times[i + 1])
            samples.append(x_batch)

            if start_idx % (batch_size * 10) == 0:
                print(f"Generated {start_idx + batch_size} samples so far")

    try:
        print("Done generating samples! Saving samples")
        samples_cartesian = final_scale * torch.cat(samples, dim=0)
        torch.save(samples_cartesian, f"{out_dir}/gen/samples_cartesian.pt")
    except Exception as e:
        breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples using the trained model")
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")

    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save generated samples")
    parser.add_argument("--num_particles", type=int, default=30, help="Number of particles in each clud")
    parser.add_argument("--num_particle_features", type=int, default=4, help="Number of features per particle")
    parser.add_argument("--final_scale", type=float, default=42.0, help="Final scale factor for the generated samples")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generating samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")

    args = parser.parse_args()
    model_info = torch.load(args.model_path, map_location=args.device)
    n_layers = 1 + int(list(model_info.keys())[-37].split("layers.")[-1].split(".")[0])
    model = JetFMGenerator(n_layers=n_layers).to(args.device)
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