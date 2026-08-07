import os

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from util.geometry.coordinates import transform_rel_particle_coordinates_to_cartesian, build_reference_vectors
from util.data import jet_attributes
from util.infra.file_management import make_clear_folder
from util.data.distributions import gen_initial_distribution


colors = sns.color_palette("deep")
import os
from util.data import jet_attributes
from util.data.distributions import gen_initial_distribution
# from util.boost_equiv import enforce_com_frame, boost_to_com_frame

from util.infra.file_management import make_clear_folder
def plot_hist(X_test_samples, x_model, output_path, filename):
    _, axs = plt.subplots(4, figsize=(10, 10))
    feature_names = [r"$E/c$", r"$p_x$", r"$p_y$", r"$p_z$"]
    for i in range(4):
        sns.histplot(
            x=X_test_samples[:, :, i].cpu().numpy().flatten(),
            ax=axs[i],
            label="Test data",
            alpha=0.75,
            color=colors[2],
            bins=100
        )
        sns.histplot(
            x=x_model[:, :, i].cpu().numpy().flatten(),
            ax=axs[i],
            label="Current data distribution",
            alpha=0.75,
            color=colors[3],
            bins=100
        )
        axs[i].set_title(f"{feature_names[i]}")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", bbox_inches='tight', dpi=300)


def generate_model_vector_field(out_dir, final_model, jet_attr_model, X_test, scale, n_jet_types, n_particles_per_jet, initial_dist_method='isotropic_com', n_features_per_particle=4, n_viz_samples=1000, zoom_in=True, save_videos=True, integration_steps=16, use_cfg=False, cfg_guidance_weight=1, use_hyperbolic=False, hyperbolic_c=1.0, use_reference_vectors=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f"{out_dir}"
    make_clear_folder(output_path)

    X_test_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_test)
    X_test_samples = X_test_particle_transformed[:n_viz_samples]

    # X_test_samples_mask = X_test_samples[..., -1]
    X_test_samples = X_test_samples[..., :-1]
    # X_test_samples = boost_to_com_frame(X_test_samples, X_test_samples_mask)

    X_test_samples = X_test_samples / scale
    X_test_samples = X_test_samples.to(device)

    print("Got x test")

    with torch.no_grad():
        final_model.eval()
        jet_attr_model.eval()
    
        generated_jet_attrs, _ = jet_attributes.generate_jets(jet_attr_model, device, n_jet_types=n_jet_types, num_jets=n_viz_samples)
        # Layout from generate_jets: [one_hot(5), eta, pt, mass, n_particles]
        jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
        gen_eta = generated_jet_attrs[:, 5].to(device)         # jet eta (raw generate_jets layout)
        gen_pt = generated_jet_attrs[:, 6].to(device)          # normalized pT from NF
        n_gen_particles = generated_jet_attrs[:, -1].long().to(device)

        n_gen_particles = torch.clamp(n_gen_particles, max=n_particles_per_jet)

        masks = jet_attributes.generate_masks(
            n_gen_particles,
            max_particles_per_jet=n_particles_per_jet,
            device=device
        )
        # Conditioning vector: [one_hot_type (5), n_particles (1), pT (1)] = 7 dims
        generated_jet_attrs = torch.cat([
            jet_one_hot_enc,
            n_gen_particles.unsqueeze(-1).float(),
            gen_pt.unsqueeze(-1),
        ], dim=-1)

        x = gen_initial_distribution(
            batch_size=n_viz_samples,
            num_particles=n_particles_per_jet,
            jet_features=generated_jet_attrs,
            prior_dist=initial_dist_method,
            device=device
        )
        x = x * masks.unsqueeze(-1).expand(-1, -1, n_features_per_particle)
        print(f"Applied masks! {x.mean()=}, {x.std()=}")

        # Symmetry-breaking reference 4-vectors (e_t + reconstructed jet p4), built from the
        # sampled jet attrs since there are no true constituents here. Mirrors generate_samples.
        ref_vectors = None
        if use_reference_vectors:
            ref_vectors = build_reference_vectors(gen_eta, gen_pt, scale, device)

        x_0 = x.clone()
        x_model_final = x.clone()

        times = torch.linspace(0, 1, integration_steps + 1).to(device)
        sns.set_palette("deep")
        for i in range(len(times) - 1):
            t = times[i].unsqueeze(0).repeat(n_viz_samples).to(device)
            final_field_conditional = final_model.forward(x_model_final, t, generated_jet_attrs, masks, ref_vectors=ref_vectors)
            if use_cfg:
                null_vector = final_model.make_null_cond(generated_jet_attrs)
                final_field_unconditional = final_model.forward(x_model_final, t, null_vector, masks, ref_vectors=ref_vectors)
                final_field = final_field_conditional + cfg_guidance_weight * (final_field_conditional - final_field_unconditional)
            else:
                final_field = final_field_conditional
            print(f"{i=}, {times[i]=}, {times[i+1]=}, {final_field.mean()=}, {final_field.std()=}")
            plt.close()
            plt.clf()
            plt.figure(figsize=(10, 10))

            final_field_flat = final_field[:, :, :2].flatten(start_dim=0, end_dim=1)
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
                x_model_final[:, :, 0].flatten().cpu().numpy(),
                x_model_final[:, :, 1].flatten().cpu().numpy(),
                label="Current data distribution",
                alpha=0.75,
                color=colors[3],
                s=0.5
            )

            plt.quiver(
                x_model_final[:, :, 0].flatten().cpu().numpy(),
                x_model_final[:, :, 1].flatten().cpu().numpy(),
                final_field_flat[:, 0].cpu().numpy(),
                final_field_flat[:, 1].cpu().numpy(),
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
            
            dt = times[i+1] - times[i]
            dt = dt.to(device)
            final_field = final_field.to(device)
            x_model_final = x_model_final + (final_field * dt)
            # x_model_final = enforce_com_frame(x_model_final, masks).to(device)

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
        x_model_final[:, :, 0].flatten().cpu().numpy(),
        x_model_final[:, :, 1].flatten().cpu().numpy(),
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
        try:
            os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_zoomed_%01d.png -vcodec mpeg4 -y  {output_path}/movie_zoomed_in.mp4")
            os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_%01d.png -vcodec mpeg4 -y  {output_path}/movie_zoomed_out.mp4")
        except Exception as e:
            print(f"Could not create videos due to error: {e}")


    plot_hist(X_test_samples, x_model_final, output_path=output_path, filename="final_hist")
    plot_hist(X_test_samples, x_0, output_path=output_path, filename="initial_hist")
    return x_0, x_model_final, masks