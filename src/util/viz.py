import os

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.Week7EGNN import JetFlowMatcher
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.jet_attributes import generate_jets, generate_masks, load_model
from util.file_management import make_clear_folder
from util.distributions import gen_initial_distribution

colors = sns.color_palette("deep")

def generate_model_vector_field(out_dir, final_model, jet_attr_model, X_test, scale, n_jet_types, 
                              n_particles_per_jet, n_features_per_particle=4, n_viz_samples=1000, 
                              zoom_in=True, save_videos=True, clamp_stddevs=3, integration_steps=16, 
                              batch_size=25):
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

        x_all = gen_initial_distribution(
            current_batch_size=n_viz_samples,
            num_particles=n_particles_per_jet,
            num_particle_features=n_features_per_particle,
            clamp_stddevs=clamp_stddevs
        ).to(device)

        x_all = x_all * masks.unsqueeze(-1).expand(-1, -1, n_features_per_particle)
        
        # Store trajectories for all samples
        trajectories = [x_all.clone().cpu()]  # Store initial state
        
        times = torch.linspace(0, 1, integration_steps + 1).to(device)
        sns.set_palette("deep")
        
        # Process each time step in batches
        for time_idx in range(len(times) - 1):
            t_current = times[time_idx]
            dt = times[time_idx + 1] - times[time_idx]
            
            # Process all samples in batches
            all_fields = []
            for batch_start in range(0, n_viz_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_viz_samples)
                
                # Extract batch
                x_batch = x_all[batch_start:batch_end]
                jet_attrs_batch = generated_jet_attrs[batch_start:batch_end]
                masks_batch = masks[batch_start:batch_end]
                t_batch = t_current.unsqueeze(0).repeat(batch_end - batch_start)
                
                # Compute vector field for batch
                field_batch = final_model.forward(x_batch, t_batch, jet_attrs_batch, masks_batch)
                all_fields.append(field_batch)
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all batch results
            final_field = torch.cat(all_fields, dim=0)
            
            print(f"Step {time_idx}: t={t_current:.3f} → {times[time_idx+1]:.3f}, "
                  f"field_mean={final_field.mean():.6f}, field_std={final_field.std():.6f}")
            
            # Update all positions
            x_all = x_all + final_field * dt
            
            # Store trajectory
            trajectories.append(x_all.clone().cpu())
            
            # Create visualization for this time step
            plt.figure(figsize=(12, 10))
            
            # Plot test data (reference)
            test_flat = X_test_samples[:, :, :2].flatten(start_dim=0, end_dim=1).cpu().numpy()
            plt.scatter(test_flat[:, 0], test_flat[:, 1], 
                       label="Test data", alpha=0.6, color=colors[2], s=0.3)
            
            # Plot current model distribution
            current_flat = x_all[:, :, :2].flatten(start_dim=0, end_dim=1).cpu().numpy()
            plt.scatter(current_flat[:, 0], current_flat[:, 1],
                       label="Current model distribution", alpha=0.7, color=colors[3], s=0.8)
            
            # Plot vector field (subsample for readability)
            field_flat = final_field[:, :, :2].flatten(start_dim=0, end_dim=1).cpu().numpy()
            field_scaled = field_flat * dt.cpu().numpy()
            
            # Subsample points for vector plotting to avoid overcrowding
            n_vectors = min(5000, len(current_flat))
            indices = np.random.choice(len(current_flat), n_vectors, replace=False)
            
            plt.quiver(current_flat[indices, 0], current_flat[indices, 1],
                      field_scaled[indices, 0], field_scaled[indices, 1],
                      label="Vector field", alpha=0.7, color=colors[0],
                      angles='xy', scale_units='xy', scale=1, width=0.002)
            
            plt.xlabel(r"$E/c$")
            plt.ylabel(r"$p_x$")
            plt.title(f"Flow Evolution: t={t_current:.3f} → t={times[time_idx+1]:.3f}")
            plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            
            # Save full view
            plt.savefig(f"{output_path}/field_vectors_{time_idx}.png", 
                       bbox_inches='tight', dpi=300)
            
            # Save zoomed view
            if zoom_in:
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.savefig(f"{output_path}/field_vectors_zoomed_{time_idx}.png", 
                           bbox_inches='tight', dpi=300)
            
            plt.close()
            
            # Clear memory
            del final_field, all_fields
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create final comparison plot
        plt.figure(figsize=(12, 10))
        
        test_flat = X_test_samples[:, :, :2].flatten(start_dim=0, end_dim=1).cpu().numpy()
        final_flat = x_all[:, :, :2].flatten(start_dim=0, end_dim=1).cpu().numpy()
        
        plt.scatter(test_flat[:, 0], test_flat[:, 1], 
                   label="Test data", alpha=0.6, color=colors[2], s=0.3)
        plt.scatter(final_flat[:, 0], final_flat[:, 1],
                   label="Final model distribution", alpha=0.7, color=colors[3], s=0.8)
        
        plt.xlabel(r"$E/c$")
        plt.ylabel(r"$p_x$")
        plt.title("Final Distribution Comparison")
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.savefig(f"{output_path}/field_vectors_{len(times)-1}.png", 
                   bbox_inches='tight', dpi=300)
        
        if zoom_in:
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.savefig(f"{output_path}/field_vectors_zoomed_{len(times)-1}.png", 
                       bbox_inches='tight', dpi=300)
        plt.close()

        # Create videos if requested
        if save_videos:
            try:
                os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_zoomed_%01d.png "
                         f"-vcodec mpeg4 -y {output_path}/movie_zoomed_in.mp4")
                os.system(f"ffmpeg -r 7 -i {output_path}/field_vectors_%01d.png "
                         f"-vcodec mpeg4 -y {output_path}/movie_zoomed_out.mp4")
            except:
                print("Warning: ffmpeg not available for video creation")

        # Create feature histograms comparison
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        feature_names = [r"$E/c$", r"$p_x$", r"$p_y$", r"$p_z$"]
        
        for i in range(4):
            ax = axs[i // 2, i % 2]
            
            # Test data histogram
            test_feature = X_test_samples[:, :, i].flatten().cpu().numpy()
            ax.hist(test_feature, bins=100, alpha=0.7, color=colors[2], 
                   label="Test data", density=True)
            
            # Final model histogram
            final_feature = x_all[:, :, i].flatten().cpu().numpy()
            ax.hist(final_feature, bins=100, alpha=0.7, color=colors[3], 
                   label="Final model", density=True)
            
            ax.set_title(f"{feature_names[i]} Distribution")
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel("Density")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/feature_histograms.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Visualization complete. Files saved to {output_path}")
        
        # Return trajectories for potential further analysis
        return torch.stack(trajectories, dim=0)  # Shape: [time_steps, n_samples, n_particles, n_features]