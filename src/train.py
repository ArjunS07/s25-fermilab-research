from copyreg import pickle
import datetime
import argparse

import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import anderson
import matplotlib.pyplot as plt
import seaborn as sns

from jetnet.datasets import JetNet
from jetnet.utils import cartesian_to_EtaPhiPtE

from models.ConditionalLEFlowMatching import JetFMGenerator
from util import jet_attributes
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.file_management import make_clear_folder
from util.distributions import sample_massless_4momentum_clouds

RANDOM_SEED = 42

MASK = False
NUM_PARTICLES = 30
NUM_PARTICLE_FEATURES = 4 # E/c, px, py, pz
TRAIN_SPLIT = 0.7


data_args = {
    "jet_type": ["g"],
    "data_dir": "datasets/jetnet",
    "num_particles": NUM_PARTICLES,
    "particle_features": (
        JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1]
    ),
    "jet_features": ["eta", "pt", "mass", "num_particles", "type"],
    "split_fraction": [TRAIN_SPLIT, 1 - TRAIN_SPLIT, 0],
    "download": True
}

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # JetNet data download args
    parser = argparse.ArgumentParser(description="Train LorentzFMNet on JetNet dataset")
    parser.add_argument("--out_dir", default="/mnt/data/output")

    parser.add_argument("--jet_types", type=str, nargs="+", default=data_args["jet_type"],
                        help="List of jet types to train on (e.g., 'g', 'q', 't')")
    parser.add_argument("--data_dir", type=str, default=data_args["data_dir"],
                        help="Directory to store the JetNet dataset")
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"],
                        help="Number of particles to consider in each jet")
    parser.add_argument("--split_fraction", type=float, nargs=3, default=data_args["split_fraction"],
                        help="Fraction of data to use for train, validation, and test splits")
    
    
    # Network hyperparameters
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in the network")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--c_weight", type=float, default=1.0, help="Weight for the c parameter in the network")
    
    # Training
    parser.add_argument("--n_train_samples", type=int, default=1000_000, help="Number of training samples to generate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train the model")

    # Integration
    parser.add_argument("--n_samples", type=int, default=50_000, help="Number of samples to generate during inference")
    parser.add_argument("--integration_steps", type=int, default=16, help="Number of integration steps for ODE solver")
    
    args = parser.parse_args()

    # Make folders if they do not exist
    out_dir = f"{args.out_dir}/{str(datetime.datetime.now()).replace(' ', '_')}"
    make_clear_folder(out_dir)
    make_clear_folder(f"{out_dir}/models")
    make_clear_folder(f"{out_dir}/logs")
    make_clear_folder(f"{out_dir}/gen")

    X_train = JetNet(
        jet_type=args.jet_types,
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        particle_features=JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1],
        jet_features=data_args["jet_features"],
        split_fraction=args.split_fraction,
        split="train",
        download=True
    )
    X_test = JetNet(
        jet_type=args.jet_types,
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        particle_features=JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1],
        jet_features=data_args["jet_features"],
        split_fraction=args.split_fraction,
        split="valid",
        download=True
    )
    print(f"{len(X_train)=}, {len(X_test)=}")
    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train)

    e_c = np.array(X_train_particle_transformed[:, :, 0].flatten())
    e_c_mirrored = np.concatenate([e_c, -e_c])
    p_x = np.array(X_train_particle_transformed[:, :, 1].flatten())
    p_y = np.array(X_train_particle_transformed[:, :, 2].flatten())
    p_z = np.array(X_train_particle_transformed[:, :, 3].flatten())
    final_scale = min([anderson(data).fit_result.params.scale for data in [e_c_mirrored, p_x, p_y, p_z]])
    with open(f"{out_dir}/logs/final_scale.txt", "w") as f:
        f.write(f"{final_scale}\n")
        f.close()
    X_train_particle_transformed = (1/final_scale) * X_train_particle_transformed

        
    model: JetFMGenerator = JetFMGenerator(
        n_particles=args.num_particles,
        n_layers=args.n_layers,
        c=args.c_weight
    ).to(device)
    torch.save(model.state_dict(), f"{out_dir}/models/model_initial.pth")

    jet_info = X_train[:][1].to(device)
    encoded_jet_types = jet_attributes.one_hot_enc_jet_type(jet_info[:, 4].long()).to(device)
    jet_info_cropped = jet_info[:, :4]  # Keep only the first 4 features (eta, p_t, mass, num_particles)
    jet_info = torch.cat([jet_info_cropped, encoded_jet_types], dim=-1).to(device)

    X_train_loaded = DataLoader(
        list(zip(X_train_particle_transformed, jet_info))[:args.n_train_samples],
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    losses = []
    # TODO: Cosine learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=16e-4)
    for epoch in range(args.num_epochs):
        epoch_loss = []

        for i, data in enumerate(X_train_loaded):
            optimizer.zero_grad()

            jet_info = data[1].to(device)
            x_1 = data[0].to(device)[:, :, :4]
            x_0 = torch.randn_like(x_1, device=device)  # Sample random initial state

            t = torch.rand(x_0.shape[0], device=device)
            t_viewed = t.view(-1, 1, 1)
            x_t = (1 - t_viewed) * x_0 + t_viewed * x_1  # Linear interpolation
            dx_t = x_1 - x_0

            pred = model.forward(x_t, t, jet_info, device=device)
            loss = nn.MSELoss()(pred, dx_t)
            loss.backward()
            model.clip_gradients()
            optimizer.step()

            epoch_loss.append(loss.item())
        
            if i % 5_000 == 0:
                print(f"dx_t: mean={dx_t.abs().mean()}, std={dx_t.abs().std()}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2       # Tensors currently live
                    reserved = torch.cuda.memory_reserved() / 1024**2         # Memory reserved by PyTorch's caching allocator
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # Peak allocation during program
                    print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}. Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Peak: {max_allocated:.2f} MB")

        losses.append(np.mean(epoch_loss))
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {losses[-1]:.4f}")
    
    with open(f"{out_dir}/logs/training_loss.csv", "w") as f:
        f.write("epoch,loss\n")
        for epoch, loss in enumerate(losses):
            f.write(f"{epoch},{loss}\n")

    torch.save(model.state_dict(), f"{out_dir}/models/final_model.pth")
    with open(f"{out_dir}/logs/args.txt", "w") as f:
        f.write(f"CLI args:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.close()

    samples = []
    jet_attr_generator = jet_attributes.load_model().to(device)
    with torch.no_grad():
        model.eval()
        jet_attr_generator.eval()
        
        times = torch.linspace(0, 1, args.integration_steps + 1).to(device)
        x = torch.randn((args.n_samples, NUM_PARTICLES, NUM_PARTICLE_FEATURES), device=device)

        for start_idx in range(0, args.n_samples, args.batch_size):
            end_idx = min(start_idx + args.batch_size, args.n_samples)
            x_batch = x[start_idx:end_idx]
            generated_jet_attrs, _ = jet_attributes.generate_jets(jet_attr_generator, device, n_jet_types=3, num_jets=args.batch_size)

            for i in range(args.integration_steps):
                x_batch = model.step(x_batch, generated_jet_attrs, times[i], times[i + 1])
            samples.append(x_batch)

    print("Done generating samples! Saving samples")
    samples_cartesian = final_scale * torch.cat(samples, dim=0)
    torch.save(samples_cartesian, f"{out_dir}/gen/samples_cartesian.pt")

    
    # Metrics
    eval_info = {}

    gen_features_absolute = cartesian_to_EtaPhiPtE(samples_cartesian)
    jet_eta = (X_test[:][1][:, 0]).unsqueeze(1)
    jet_phi_vals = (2 * torch.pi) * torch.rand(len(X_test)).unsqueeze(1)
    jet_phi_vals -= torch.pi
    jet_pt_ec = X_test[:][1][:, 1:3]
    jet_features = torch.concat([jet_eta, jet_phi_vals, jet_pt_ec], dim=-1)
    eta_rel, phi_rel, pt_rel = torch.unbind(X_test[:][0][:, :, :3], axis=-1)
    Eta, Phi, Pt, _ = torch.unbind(jet_features, axis=-1)
    pt = pt_rel * Pt.unsqueeze(1)
    eta = eta_rel + Eta.unsqueeze(1)
    phi = phi_rel + Phi.unsqueeze(1)
    p0 = pt * torch.cosh(eta)
    test_features_absolute = [eta, phi, pt, p0]

    eval_info["cov_mmd"] = eval.cov_mmd(
        real_jets=test_features_absolute[:, :, :3].to(device),
        gen_jets=gen_features_absolute
    )
    eval_info["fpd"] = eval.fpd(
        real_features=test_features_absolute.reshape((-1, NUM_PARTICLE_FEATURES)).to(device),
        gen_features=gen_features_absolute.reshape((-1, NUM_PARTICLE_FEATURES)).to(device),
        seed=RANDOM_SEED
    )
    jets1 = gen_features_absolute[:, :, :]
    jets2 = test_features_absolute[:][0]
    try:
        eval_info["w1efp"] = eval.w1efp(
            jets1=jets1,
            jets2=jets2,
        )
    except Exception as e:
        print(f"Error calculating W1 EFP metrics: {e}")
    try:
        eval_info["w1m"] = eval.w1m(
            jets1=jets1,
            jets2=jets2,
        )
        eval_info["w1p"] = eval.w1p(
            jets1=jets1,
            jets2=jets2,
        )
    except Exception as e:
        print(f"Error calculating W1 metrics: {e}")

    with open(f"{out_dir}/logs/eval_info.pkl", "wb") as f:
        pickle.dump(eval_info, f)
    print("Evaluation metrics saved to eval_info.pkl")

    # Generate figures
    features_polar = [r"$\eta$", r"$\phi$", r"$p_T$", r"$E/c$"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    for i, feature in enumerate(features_polar):
        sns.kdeplot(
            x=gen_features_absolute[:, :, i].flatten().cpu().numpy(),
            ax=axs[i],
            label="Generated",
            fill=True,
            alpha=0.5
        )
        sns.kdeplot(
            x=test_features_absolute[:, :, i].flatten().cpu().numpy(),
            ax=axs[i],
            label="Test set",
            fill=True,
            alpha=0.5
        )
        axs[i].set_title(feature)
        axs[i].legend()
    plt.suptitle("Feature Distributions: Generated vs Real Jets")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gen/feature_distributions.png", dpi=300)
