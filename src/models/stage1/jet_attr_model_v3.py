"""Hybrid Stage-1 model: categorical multiplicity plus conditional spline flow.

Multiplicity is fitted exactly from empirical counts.  Conditional on jet type and
multiplicity, a rational-quadratic spline flow models
``(eta, log(pT), log(mass / pT))``.
"""

from __future__ import annotations

import argparse
import json
import pickle

import normflows as nf
import torch
import torch.nn as nn
import torch.nn.functional as F


class JetAttributeFlowV3(nn.Module):
    def __init__(
        self, max_particles=150, num_types=5, num_flows=10,
        hidden_layers=4, hidden_units=128,
    ):
        super().__init__()
        self.max_particles = max_particles
        self.num_types = num_types
        self.num_flows = num_flows
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        context_size = num_types + 1
        flows = []
        for _ in range(num_flows):
            flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(
                3, hidden_layers, hidden_units, num_context_channels=context_size
            ))
            flows.append(nf.flows.LULinearPermute(3))
        self.flow = nf.ConditionalNormalizingFlow(
            nf.distributions.DiagGaussian(3, trainable=False), flows
        )
        self.register_buffer(
            "multiplicity_probs", torch.ones(num_types, max_particles + 1)
        )
        self.register_buffer("feature_mean", torch.zeros(3))
        self.register_buffer("feature_std", torch.ones(3))

    def flow_context(self, context, multiplicity):
        scaled_n = (multiplicity.float() / self.max_particles).unsqueeze(-1)
        return torch.cat([context.float(), scaled_n], dim=-1)

    def transformed_features(self, attrs):
        pt = attrs[:, 1].clamp(min=1e-8)
        ratio = attrs[:, 2].clamp(min=1e-8) / pt
        values = torch.stack([attrs[:, 0], pt.log(), ratio.log()], dim=-1)
        return (values - self.feature_mean) / self.feature_std

    def nll(self, attrs, context):
        multiplicity = attrs[:, 3].long().clamp(0, self.max_particles)
        values = self.transformed_features(attrs)
        return self.flow.forward_kld(values, self.flow_context(context, multiplicity))

    @torch.no_grad()
    def sample(self, num_samples, context):
        type_ids = context.argmax(-1)
        multiplicity = torch.multinomial(
            self.multiplicity_probs[type_ids], 1, replacement=True
        ).squeeze(-1)
        values, log_prob = self.flow.sample(
            num_samples, context=self.flow_context(context, multiplicity)
        )
        values = values * self.feature_std + self.feature_mean
        pt = values[:, 1].exp()
        mass = pt * values[:, 2].exp()
        attrs = torch.stack(
            [values[:, 0], pt, mass, multiplicity.float()], dim=-1
        )
        return attrs, log_prob

    def portable_config(self):
        return {
            "max_particles": self.max_particles,
            "num_types": self.num_types,
            "num_flows": self.num_flows,
            "hidden_layers": self.hidden_layers,
            "hidden_units": self.hidden_units,
        }


def _features(attrs):
    pt = attrs[:, 1].clamp(min=1e-8)
    return torch.stack([
        attrs[:, 0], pt.log(), (attrs[:, 2].clamp(min=1e-8) / pt).log()
    ], dim=-1)


def _fit_multiplicity(model, attrs, type_ids):
    counts = torch.zeros_like(model.multiplicity_probs)
    counts.index_put_(
        (type_ids, attrs[:, 3].long().clamp(0, model.max_particles)),
        torch.ones(len(attrs), device=attrs.device),
        accumulate=True,
    )
    empty = counts.sum(-1) == 0
    counts[empty, model.max_particles] = 1.0
    model.multiplicity_probs.copy_(counts / counts.sum(-1, keepdim=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="/mnt/data/output")
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_particles", type=int, default=150)
    parser.add_argument("--num_flows", type=int, default=10)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--validation_interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)
    with open(f"{args.output_path}/data/x_train.pkl", "rb") as handle:
        dataset = pickle.load(handle)
    attrs = dataset[:][1].float()
    permutation = torch.randperm(len(attrs))
    n_valid = max(1, int(0.1 * len(attrs)))
    valid_attrs = attrs[permutation[:n_valid]]
    train_attrs = attrs[permutation[n_valid:]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_attrs, valid_attrs = train_attrs.to(device), valid_attrs.to(device)
    train_types = train_attrs[:, -1].long()
    valid_types = valid_attrs[:, -1].long()
    train_context = F.one_hot(train_types, num_classes=5).float()
    valid_context = F.one_hot(valid_types, num_classes=5).float()

    model = JetAttributeFlowV3(
        max_particles=args.max_particles, num_flows=args.num_flows,
        hidden_layers=args.hidden_layers, hidden_units=args.hidden_units,
    ).to(device)
    transformed = _features(train_attrs)
    model.feature_mean.copy_(transformed.mean(0))
    model.feature_std.copy_(transformed.std(0).clamp(min=1e-6))
    _fit_multiplicity(model, train_attrs, train_types)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_validation = float("inf")
    best_state = None
    stale = 0
    history = []
    for step in range(1, args.max_steps + 1):
        index = torch.randint(
            len(train_attrs), (min(args.batch_size, len(train_attrs)),), device=device
        )
        loss = model.nll(train_attrs[index, :4], train_context[index])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % args.validation_interval == 0 or step == 1:
            with torch.no_grad():
                validation_chunks = []
                for start in range(0, len(valid_attrs), args.batch_size):
                    validation_chunks.append(float(model.nll(
                        valid_attrs[start:start + args.batch_size, :4],
                        valid_context[start:start + args.batch_size],
                    )))
                validation = float(sum(validation_chunks) / len(validation_chunks))
            history.append({
                "step": step, "train_nll": float(loss), "validation_nll": validation
            })
            print(
                f"Stage1-v3 step {step}/{args.max_steps}: "
                f"train={float(loss):.5f} validation={validation:.5f}",
                flush=True,
            )
            if validation < best_validation:
                best_validation = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping at step {step}", flush=True)
                    break
    if best_state is None:
        raise RuntimeError("Stage-1 v3 never produced a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval().cpu()
    torch.save({
        "format": "jet_attribute_v3_state_dict",
        "config": model.portable_config(),
        "state_dict": model.state_dict(),
        "training": {
            "best_validation_nll": best_validation,
            "history": history,
            "coordinate_system": ["eta", "log_pt", "log_mass_over_pt"],
        },
    }, f"{args.output_path}/jet_attr_model.pth")
    with open(f"{args.output_path}/stage1_v3_training.json", "w") as handle:
        json.dump({
            "best_validation_nll": best_validation, "history": history,
            "coordinate_system": ["eta", "log_pt", "log_mass_over_pt"],
        }, handle, indent=2)


if __name__ == "__main__":
    main()
