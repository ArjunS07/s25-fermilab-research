import argparse
import pickle
import os

import numpy as np
import torch
import normflows as nf
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_data_path
from util.jet_attributes import one_hot_enc_jet_type, NUM_CLASSES
from util.file_management import make_clear_folder
from jetnet.evaluation import fpd, kpd

RANDOM_SEED = 42
JET_FEATURES = [r"$\eta$", r"$p_T$", r"$m$", r"$N_P$"]
jet_type_map = {
    0: "Gluons",
    1: "Light quarks",
    2: "Top quarks",
}

def get_model_pth_path(output_path):
    return os.path.join(output_path, "jet_attr_model.pth")

def noise_num_particles(X, noise_std=0.15):
    """
    Add noise to the number of particles in the jets.
    
    Args:
        X (torch.Tensor): Input tensor with jet features.
        noise_std (float): Standard deviation of the noise to be added.
        
    Returns:
        torch.Tensor: Tensor with noisy number of particles.
    """
    num_particles = X[:, 3]
    noise = torch.randn_like(num_particles) * noise_std
    noise = torch.clamp(noise, -1, 1)
    return num_particles + noise

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)

    plt.rc("mathtext", fontset="cm") 
    sns.set_palette("deep")

    parser = argparse.ArgumentParser(description="Train Jet Attribute NF Model on JetNet dataset")
    parser.add_argument("--output_path", type=str, default="/mnt/data/output", help="Path to save the output files")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=35_000, help="Number of epochs to train the model")
    parser.add_argument("--K", type=int, default=10, help="Number of layers in the flow")
    parser.add_argument("--hidden_units", type=int, default=128, help="Number of hidden units in the flow")
    parser.add_argument("--hidden_layers", type=int, default=8, help="Number of hidden layers in the flow")
    parser.add_argument("--n_test_samples", type=int, default=50_000, help="Number of test samples to generate")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the trained model")

    args = parser.parse_args()
    data_path = get_data_path(args.output_path)
    jet_attr_model_path = f"{args.output_path}/jet_attr_model"
    make_clear_folder(jet_attr_model_path)
    
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(f"{data_path}/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    # Only take global jet features
    X_train = X_train[:][1]
    X_test = X_test[:][1]

    # equalize the number of jets per type
    X_train_equalized = []
    jet_types_ints = set(X_train[:, -1].long().tolist())
    min_jets_per_type = min(len(X_train[X_train[:, -1] == jet_type]) for jet_type in jet_types_ints)
    for jet_type in jet_types_ints:
        X_train_equalized.extend(X_train[X_train[:, -1] == jet_type][:min_jets_per_type])
    X_train = torch.tensor(np.array(X_train_equalized), dtype=torch.float32)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_bins = 200
    std = 0.15
    num_particles = X_train[:, 3]
    noised_particles = noise_num_particles(X_train, noise_std=std)
    X_train[:, 3] = noised_particles

    X_train = X_train.to(device)

    long_types = X_train[:, -1].long()
    one_hot_jets = one_hot_enc_jet_type(long_types)
    print(one_hot_jets.shape, one_hot_jets.sum(dim=0))

    K = args.K
    hidden_units = args.hidden_units
    hidden_layers = args.hidden_layers

    # Fixed
    latent_size = 4
    context_size = NUM_CLASSES

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    model = nf.ConditionalNormalizingFlow(q0, flows).to(device)

    torch.manual_seed(RANDOM_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    epochs = args.num_epochs
    batch_size = args.batch_size

    loss_hist = np.array([])
    for it in range(epochs):
        optimizer.zero_grad()

        indices = torch.randperm(len(X_train), device=device)[:batch_size]
        jets = X_train[indices]
        jet_info = jets[:, :-1]  # Exclude the jet type column
        jet_types = jets[:, -1].long()  # Get the jet type column
        one_hot_types = one_hot_enc_jet_type(jet_types).to(device)
        
        loss = model.forward_kld(jet_info, one_hot_types)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    np.savetxt(f"{jet_attr_model_path}/jet_attr_model_loss.csv", loss_hist, delimiter=",")
    if args.save_model:
        with open(get_model_pth_path(args.output_path), "wb") as f:
            torch.save(model, f)
    
    sns.set_palette("deep")
    sns.lineplot(
        x=np.arange(len(loss_hist)),
        y=loss_hist,
        label="Loss",
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Forward KL Divergence Loss", fontsize=14)
    plt.savefig(f"{jet_attr_model_path}/jet_attr_model_loss_hist.png", dpi=300, bbox_inches="tight")
    
    model.eval()
    n_samples = args.n_test_samples
    sample_jet_types = torch.randint(0, NUM_CLASSES, (n_samples,)).to(device)
    one_hot_types = one_hot_enc_jet_type(sample_jet_types)
    sample_vals, sample_logprobs = model.sample(n_samples, context=one_hot_types)
    sample_jet_types.shape, sample_vals.shape, sample_logprobs.shape
    # Quantize the number of particles
    sample_vals[:, 3] = torch.round(sample_vals[:, 3])

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for i, feature in enumerate(JET_FEATURES):
        ax = axs[i]
        sns.kdeplot(
            sample_vals[:, i].detach().numpy(),
            ax=ax,
            label=r"Generated " + feature + r" (all jets)",
        )
        sns.kdeplot(
            X_test[:, i].numpy(),
            ax=ax,
            label=r"Test " + feature + r" (all jets)",
        )
        ax.set_xlabel(feature, fontsize=18)
        if i == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
        ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{jet_attr_model_path}/jet_attr_nf_sampled_all_features.png", dpi=300, bbox_inches="tight")

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    for i, feature in enumerate(JET_FEATURES):
        ax = axs[0, i]
        for jet_type in jet_type_map.keys():
            sns.kdeplot(
                X_test[X_test[:, -1] == jet_type][:, i].numpy(),
                ax=ax,
                label=jet_type_map[jet_type]
            )
        if i == 0:
            ax.set_ylabel("Test set jet attributes", fontsize=14, rotation=0, labelpad=60)
        else:
            ax.set_ylabel("")

        ax = axs[1, i]
        for jet_type in jet_type_map.keys():
            sns.kdeplot(
                sample_vals[sample_jet_types == jet_type][:, i].detach().numpy(),
                ax=ax,
                label=jet_type_map[jet_type]
            )
        ax.set_xlabel(feature, fontsize=18)
        if i == 0:
            ax.set_ylabel("Generated jet attributes", fontsize=14, rotation=0, labelpad=60)
        else:
            ax.set_ylabel("")

    for i in range(4):
        ax1 = axs[0, i]
        ax2 = axs[1, i]

        xlims = np.array([ax1.get_xlim(), ax2.get_xlim()])
        ylims = np.array([ax1.get_ylim(), ax2.get_ylim()])
        xlim_combined = (xlims[:, 0].min(), xlims[:, 1].max())
        ylim_combined = (ylims[:, 0].min(), ylims[:, 1].max())

        ax1.set_xlim(xlim_combined)
        ax2.set_xlim(xlim_combined)
        ax1.set_ylim(ylim_combined)
        ax2.set_ylim(ylim_combined)


    legend_handles, legend_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        title="",
        loc="center right",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=14,
        title_fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"{jet_attr_model_path}/jet_attr_nf_sampled_jet_type.png", dpi=300, bbox_inches="tight")

    try:

        fpd_val, fpd_err = fpd(
            real_features=X_test[:, :-1].numpy(),
            gen_features=sample_vals.detach().numpy(),
            seed=RANDOM_SEED
        )
        kpd_val, kpd_err = kpd(
            real_features=X_test[:, :-1].numpy(),
            gen_features=sample_vals.detach().numpy(),
            seed=RANDOM_SEED
        )
        with open(f"{jet_attr_model_path}/jet_attr_nf_metrics.txt", "w") as f:
            f.write(f"FPD: {fpd_val:.5E} ± {fpd_err:.5E}\n")
            f.write(f"KPD: {kpd_val:.5E} ± {kpd_err:.5E}\n")
    except Exception as e:
        print(f"Error occurred while computing metrics: {e}")

    with open(f"{jet_attr_model_path}/jet_attr_model_info.txt", "w") as f:
        f.write(f"CLI args:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.close()