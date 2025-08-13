import pickle
import os
import argparse

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models.Week7EGNN import JetFlowMatcher
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.jet_attributes import generate_jets, generate_masks, load_model
from util.file_management import make_clear_folder
from util.distributions import gen_initial_distribution

colors = sns.color_palette("deep")
def generate_model_vector_field(out_dir, final_model, jet_attr_model, X_test, scale, n_jet_types, n_particles_per_jet, n_features_per_particle=4, n_viz_samples=1000, zoom_in=True, save_videos=True, clamp_stddevs=3, integration_steps=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f"{out_dir}/vf_viz"
    make_clear_folder(output_path)

    X_test_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_test)
    X_test_samples = X_test_particle_transformed[:n_viz_samples]
    X_test_samples = X_test_samples / scale
    X_test_samples = X_test_samples.to(device)

    with torch.no_grad():
        final_model.eval()
        jet_attr_model.eval()
    
        generated_jet_attrs, _ = generate_jets(jet_attr_model, device, n_jet_types=n_jet_types, num_jets=n_viz_samples)
        jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
        n_gen_particles = generated_jet_attrs[:, -1].long().to(device)

        masks = generate_masks(
            n_gen_particles,
            max_particles_per_jet=n_particles_per_jet,
            device=device
        )
        generated_jet_attrs = torch.cat([
            jet_one_hot_enc,
            n_gen_particles.unsqueeze(-1).float()
        ], dim=-1)

        x = gen_initial_distribution(
            current_batch_size=n_viz_samples,
            num_particles=n_particles_per_jet,
            num_particle_features=n_features_per_particle,
            clamp_stddevs=clamp_stddevs
    )
        x = x.to(device)

        x = x * masks.unsqueeze(-1).expand(-1, -1, n_features_per_particle)
        x_model_final = x.clone()

        times = torch.linspace(0, 1, integration_steps + 1).to(device)
        sns.set_palette("deep")
        for i in range(len(times) - 1):
            t = times[i].unsqueeze(0).repeat(n_viz_samples).to(device)
            final_field = final_model.forward(x_model_final, t, generated_jet_attrs, masks)
            print(f"{i=}, {times[i]=}, {times[i+1]=}, {final_field.mean()=}, {final_field.std()=}")
            plt.clf()

            final_field_flat = final_field[:, :, :2].flatten(start_dim=0, end_dim=1)[:1000]
            final_field_flat = final_field_flat * (times[i+1] - times[i])

            sns.scatterplot(
                x=X_test_samples[:, :, 0].cpu().numpy().flatten(),
                y=X_test_samples[:, :, 1].cpu().numpy().flatten(),
                label="Test data",
                alpha=0.75,
                color=colors[2],
                s=0.2
            )
            plt.scatter(
                x_model_final[:, :, 0].flatten()[:1000].cpu().numpy(),
                x_model_final[:, :, 1].flatten()[:1000].cpu().numpy(),
                label="Current data distribution",
                alpha=0.75,
                color=colors[3],
                s=0.5
            )
            plt.quiver(
                x_model_final[:, :, 0].flatten()[:1000].cpu().numpy(),
                x_model_final[:, :, 1].flatten()[:1000].cpu().numpy(),
                final_field_flat[:, 0],
                final_field_flat[:, 1],
                label="Final model vector field",
                alpha=0.75,
                color=colors[0]
            )

            plt.xlabel(r"$E/c$ ")
            plt.ylabel(r"$p_x$")
            plt.title(r"$t=%.2f \rightarrow t=%.2f$" % (times[i].item(), times[i+1].item()))

            plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            plt.savefig(f"{output_path}/field_vectors_{i}.png", bbox_inches='tight', dpi=300)

            if zoom_in:
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.savefig(f"{output_path}/field_vectors_zoomed_{i}.png", bbox_inches='tight', dpi=300)
            
            print(f"{x_model_final.mean()=}, {x_model_final.std()=}")
            x_model_final = x_model_final + (final_field * (times[i+1] - times[i]))
            print(f"{x_model_final.mean()=}, {x_model_final.std()=}")

            plt.clf()
            plt.close()
            del final_field, final_field_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final status
    sns.scatterplot(
        x=X_test_samples[:, :, 0].cpu().numpy().flatten(),
        y=X_test_samples[:, :, 1].cpu().numpy().flatten(),
        label="Test data",
        alpha=0.75,
        color=colors[2],
        s=0.2
    )
    plt.scatter(
        x_model_final[:, :, 0].flatten()[:1000].cpu().numpy(),
        x_model_final[:, :, 1].flatten()[:1000].cpu().numpy(),
        label="Current data distribution",
        alpha=0.75,
        color=colors[3],
        s=0.5
    )        
    if save_videos:
        os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_zoomed_%01d.png -vcodec mpeg4 -y debugging/movie_zoomed_in.mp4")
        os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_%01d.png -vcodec mpeg4 -y debugging/movie_zoomed_out.mp4")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Generate vector field visualizations for the final model.")
    parser.add_argument("--path", type=str, required=True, help="Path to the downloaded output directory.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to visualize.")
    parser.add_argument("--n_jet_types", type=int, default=3)

    args = parser.parse_args()

    jet_attr_generator = load_model(f"{args.path}/jet_attr_model.pth").to(device)

    n_particles = 150
    n_features = 4

    scale = open(f"{args.path}/train/scale.txt").read().strip()
    scale = float(scale)

    with open(f"{args.path}/data/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)


    final_model_info = torch.load(f"{args.path}/train/models/final_model.pth", map_location=device, weights_only=False)
    final_model = JetFlowMatcher(
        max_num_jet_types=5, 
        num_layers=3,
        hidden_dim=128
    )

    generate_model_vector_field(
        final_model=final_model,
        jet_attr_model=jet_attr_generator,
        X_test=X_test,
        scale=scale,
        n_jet_types=args.n_jet_types,
        n_particles_per_jet=n_particles,
        n_features_per_particle=n_features,
        n_viz_samples=args.n_samples
    )