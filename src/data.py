import argparse
import os
import pickle

import torch
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear

RANDOM_SEED = 42

TRAIN_SPLIT = 0.7
MASK = True
data_args = {
    "jet_type": ["g", "q", "t"],
    "data_dir": "datasets/jetnet",
    "num_particles": 150, 
    "particle_features": (
        JetNet.ALL_PARTICLE_FEATURES if MASK else JetNet.ALL_PARTICLE_FEATURES[:-1]
    ),
    "jet_features": ["eta", "pt", "mass", "num_particles", "type"],
    "jet_normalisation": FeaturewiseLinear(),
    "split_fraction": [TRAIN_SPLIT, 1 - TRAIN_SPLIT, 0],
    "download": True
}

def get_data_path(process_id):
    return f"data/{process_id}"

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--jet_types", type=str, nargs="+", default=data_args["jet_type"],
                        help="List of jet types to train on (e.g., 'g', 'q', 't')")
    parser.add_argument("--process_id", type=str, default="abcd", help="Process ID for distributed training")

    # parser.add_argument("--data_dir", type=str, default=data_args["data_dir"],
    #                     help="Directory to store the JetNet dataset")
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"],
                        help="Number of particles to consider in each jet")
    # parser.add_argument("--split_fraction", type=float, nargs=3, default=data_args["split_fraction"],
    #                     help="Fraction of data to use for train, validation, and test splits")

    args = parser.parse_args()

    print(args.jet_types)

    X_train = JetNet(
        jet_type=args.jet_types,
        data_dir=data_args["data_dir"],
        num_particles=args.num_particles,
        particle_features=data_args["particle_features"],
        jet_features=data_args["jet_features"],
        split_fraction=data_args["split_fraction"],
        jet_normalisation=data_args["jet_normalisation"],
        split="train",
        download=True
    )
    X_test = JetNet(
        jet_type=data_args["jet_type"],
        data_dir=data_args["data_dir"],
        num_particles=args.num_particles,
        particle_features=data_args["particle_features"],
        jet_features=data_args["jet_features"],
        split_fraction=data_args["split_fraction"],
        jet_normalisation=data_args["jet_normalisation"],
        split="valid",
        download=True
    )

    data_path = get_data_path(args.process_id)
    os.makedirs(data_path, exist_ok=True)

    with open(f"{data_path}/x_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(f"{data_path}/x_test.pkl", "wb") as f:
        pickle.dump(X_test, f)