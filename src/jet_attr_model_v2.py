"""Discrete-multiplicity conditional mixture model for jet attributes.

Stage 1 v2 models N exactly as a categorical variable and models
``(eta, log pT, log mass)`` with a conditional Gaussian mixture. This avoids the legacy
noise/round/clip path and persists its feature transforms in the module state.
"""

from __future__ import annotations

import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class JetAttributeModelV2(nn.Module):
    def __init__(self, max_particles=150, num_types=5, mixtures=8, hidden=128):
        super().__init__()
        self.max_particles = max_particles
        self.num_types = num_types
        self.mixtures = mixtures
        self.net = nn.Sequential(
            nn.Linear(num_types + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, mixtures * 7),
        )
        self.register_buffer("multiplicity_probs", torch.ones(num_types, max_particles + 1))
        self.register_buffer("feature_mean", torch.zeros(3))
        self.register_buffer("feature_std", torch.ones(3))

    def parameters_for(self, context, multiplicity):
        n = (multiplicity.float() / self.max_particles).unsqueeze(-1)
        raw = self.net(torch.cat([context.float(), n], dim=-1))
        raw = raw.view(-1, self.mixtures, 7)
        logits = raw[..., 0]
        means = raw[..., 1:4]
        log_scales = raw[..., 4:7].clamp(-5.0, 3.0)
        return logits, means, log_scales

    def transformed_features(self, attrs):
        values = torch.stack([
            attrs[:, 0], attrs[:, 1].clamp(min=1e-6).log(),
            attrs[:, 2].clamp(min=1e-6).log(),
        ], dim=-1)
        return (values - self.feature_mean) / self.feature_std

    def nll(self, attrs, context):
        multiplicity = attrs[:, 3].long().clamp(0, self.max_particles)
        target = self.transformed_features(attrs).unsqueeze(1)
        logits, means, log_scales = self.parameters_for(context, multiplicity)
        component = -0.5 * (((target - means) / log_scales.exp()).square()
                            + 2 * log_scales + torch.log(torch.tensor(2 * torch.pi,
                                                                      device=attrs.device))).sum(-1)
        continuous = -torch.logsumexp(F.log_softmax(logits, -1) + component, dim=-1).mean()
        type_ids = context.argmax(-1)
        discrete = -self.multiplicity_probs[type_ids, multiplicity].clamp(min=1e-12).log().mean()
        return continuous + discrete

    @torch.no_grad()
    def sample(self, num_samples, context):
        device = context.device
        type_ids = context.argmax(-1)
        multiplicity = torch.multinomial(
            self.multiplicity_probs[type_ids], 1, replacement=True).squeeze(-1)
        logits, means, log_scales = self.parameters_for(context, multiplicity)
        component = torch.distributions.Categorical(logits=logits).sample()
        row = torch.arange(num_samples, device=device)
        z = means[row, component] + log_scales[row, component].exp() * torch.randn(
            num_samples, 3, device=device)
        values = z * self.feature_std + self.feature_mean
        attrs = torch.stack([
            values[:, 0], values[:, 1].exp(), values[:, 2].exp(), multiplicity.float()
        ], dim=-1)
        return attrs, torch.zeros(num_samples, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="/mnt/data/output")
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_particles", type=int, default=150)
    args = parser.parse_args()
    torch.manual_seed(42)
    with open(f"{args.output_path}/data/x_train.pkl", "rb") as handle:
        dataset = pickle.load(handle)
    attrs = dataset[:][1].float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attrs = attrs.to(device)
    type_ids = attrs[:, -1].long()
    context = F.one_hot(type_ids, num_classes=5).float()
    model = JetAttributeModelV2(max_particles=args.max_particles).to(device)

    transformed = torch.stack([
        attrs[:, 0], attrs[:, 1].clamp(min=1e-6).log(), attrs[:, 2].clamp(min=1e-6).log()
    ], -1)
    model.feature_mean.copy_(transformed.mean(0))
    model.feature_std.copy_(transformed.std(0).clamp(min=1e-6))
    counts = torch.zeros_like(model.multiplicity_probs)
    counts.index_put_((type_ids, attrs[:, 3].long().clamp(0, args.max_particles)),
                      torch.ones(len(attrs), device=device), accumulate=True)
    empty = counts.sum(-1) == 0
    counts[empty, args.max_particles] = 1.0
    model.multiplicity_probs.copy_(counts / counts.sum(-1, keepdim=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    for epoch in range(args.num_epochs):
        index = torch.randint(len(attrs), (min(args.batch_size, len(attrs)),), device=device)
        loss = model.nll(attrs[index, :4], context[index])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Stage1-v2 epoch {epoch}/{args.num_epochs}: nll={loss.item():.5f}", flush=True)
    model.eval().cpu()
    torch.save(model, f"{args.output_path}/jet_attr_model.pth")


if __name__ == "__main__":
    main()
