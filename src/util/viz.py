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
        masks = masks.to(device)
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
            plt.close()
            plt.clf()
            plt.figure(figsize=(10, 10))

            final_field_flat = final_field[:, :, :2].flatten(start_dim=0, end_dim=1)
            final_field_flat = final_field_flat * (times[i+1] - times[i])

            sns.scatterplot(
                x=X_test_samples[:, :, 0].detach().cpu().numpy().flatten(),
                y=X_test_samples[:, :, 1].detach().cpu().numpy().flatten(),
                label="Test data",
                alpha=0.75,
                color=colors[2],
                s=0.2
            )
            plt.scatter(
                x_model_final[:, :, 0].flatten().detach().cpu().numpy(),
                x_model_final[:, :, 1].flatten().detach().cpu().numpy(),
                label="Current data distribution",
                alpha=0.75,
                color=colors[3],
                s=0.5
            )

            plt.quiver(
                x_model_final[:, :, 0].flatten().detach().cpu().numpy(),
                x_model_final[:, :, 1].flatten().detach().cpu().numpy(),
                final_field_flat.detach().cpu().numpy()[:, 0],
                final_field_flat.detach().cpu().numpy()[:, 1],
                label="Final model vector field",
                alpha=0.75,
                color=colors[0],
                angles='xy',  # interpret vector components in data coordinates
                scale_units='xy',  # scale arrows in the same units as x/y
                scale=1           # no automatic rescaling
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
            
            x_model_final = x_model_final.detach() + (final_field.detach() * (times[i+1] - times[i]))

            plt.clf()
            plt.close()
            del final_field, final_field_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final status
    sns.scatterplot(
        x=X_test_samples[:, :, 0].detach().cpu().numpy().flatten(),
        y=X_test_samples[:, :, 1].detach().cpu().numpy().flatten(),
        label="Test data",
        alpha=0.75,
        color=colors[2],
        s=0.2
    )
    plt.scatter(
        x_model_final[:, :, 0].flatten().detach().cpu().numpy(),
        x_model_final[:, :, 1].flatten().detach().cpu().numpy(),
        label="Current data distribution",
        alpha=0.75,
        color=colors[3],
        s=0.5
    )
    plt.xlabel(r"$E/c$ ")
    plt.ylabel(r"$p_x$")
    plt.title(r"$t=%.2f \rightarrow t=%.2f$" % (times[i].item(), times[i+1].item()))
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.savefig(f"{output_path}/field_vectors_{i+1}.png", bbox_inches='tight', dpi=300)
    if zoom_in:
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig(f"{output_path}/field_vectors_zoomed_{i+1}.png", bbox_inches='tight', dpi=300)
    
    if save_videos:
        os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_zoomed_%01d.png -vcodec mpeg4 -y  {output_path}/movie_zoomed_in.mp4")
        os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_%01d.png -vcodec mpeg4 -y  {output_path}/movie_zoomed_out.mp4")

    fig, axs = plt.subplots(4, figsize=(10, 10))
    feature_names = [r"$E/c$", r"$p_x$", r"$p_y$", r"$p_z$"]
    for i in range(4):
        sns.histplot(
            x=X_test_samples[:, :, i].detach().cpu().numpy().flatten(),
            ax=axs[i],
            label="Test data",
            alpha=0.75,
            color=colors[2],
            bins=100
        )
        sns.histplot(
            x=x_model_final[:, :, i].detach().cpu().numpy().flatten(),
            ax=axs[i],
            label="Current data distribution",
            alpha=0.75,
            color=colors[3],
            bins=100
        )
        axs[i].set_title(f"{feature_names[i]}")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_histograms.png", bbox_inches='tight', dpi=300)