import torch
from util.geometry.coordinates import eta_phi_pt_e_to_cartesian

def gen_initial_distribution(x_1=None, batch_size=None, num_particles=None,
                             prior_dist='axis_aligned_per_jet', jet_features=None,
                             jet_phi=None, device='cpu', model_scale=None,
                             particle_mask=None):

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]
        device = x_1.device

    if prior_dist == 'axis_aligned_per_jet':
        # Batch-independent physical prior. Positive lognormal weights are normalized within
        # each jet, and one scalar correction makes the transverse vector sum match the
        # conditioning pT exactly. The result is then put directly in model-scaled units.
        if jet_features is None or model_scale is None:
            raise ValueError("axis_aligned_per_jet requires jet_features and model_scale")
        device = jet_features.device
        jet_eta = jet_features[:, 0]
        jet_pt = jet_features[:, 1]
        phi0 = ((2 * torch.pi) * torch.rand(batch_size, device=device)
                if jet_phi is None else jet_phi.to(device))
        eta = jet_eta.unsqueeze(1) + 0.15 * torch.randn(batch_size, num_particles, device=device)
        phi = phi0.unsqueeze(1) + 0.15 * torch.randn(batch_size, num_particles, device=device)
        weights = torch.softmax(torch.randn(batch_size, num_particles, device=device), dim=1)
        ux, uy = torch.cos(phi), torch.sin(phi)
        resultant = torch.sqrt((weights * ux).sum(1).square() + (weights * uy).sum(1).square())
        pt = weights * (jet_pt / resultant.clamp(min=1e-4)).unsqueeze(1)
        energy = pt * torch.cosh(eta)
        particles = eta_phi_pt_e_to_cartesian(torch.stack([eta, phi, pt, energy], dim=-1))
        return particles / float(model_scale)

    elif prior_dist == 'axis_aligned_equal':
        # Simple conditioned-axis prior: equal pT shares over real particles, followed by
        # one per-jet scalar correction so the transverse vector sum matches conditioned pT.
        # It deliberately carries no fragmentation hierarchy or conditioned jet mass.
        if jet_features is None or model_scale is None or particle_mask is None:
            raise ValueError(
                "axis_aligned_equal requires jet_features, model_scale, and particle_mask"
            )
        if float(model_scale) <= 0:
            raise ValueError("model_scale must be positive")
        device = jet_features.device
        real = particle_mask.to(device=device, dtype=jet_features.dtype)
        if real.shape != (batch_size, num_particles):
            raise ValueError(
                "particle_mask shape must equal (batch_size, num_particles); "
                f"got {tuple(real.shape)}"
            )
        multiplicity = real.sum(dim=1, keepdim=True)
        if (multiplicity < 1).any():
            raise ValueError("axis_aligned_equal requires at least one real particle per jet")

        jet_eta = jet_features[:, 0]
        jet_pt = jet_features[:, 1]
        phi0 = (
            (2 * torch.pi) * torch.rand(batch_size, device=device)
            if jet_phi is None else jet_phi.to(device)
        )
        # Draw both offsets together so a jet's sample is unchanged when unrelated jets
        # are appended to the batch under the same RNG seed.
        angular_offsets = 0.15 * torch.randn(
            batch_size, num_particles, 2, device=device, dtype=jet_features.dtype
        )
        eta = jet_eta.unsqueeze(1) + angular_offsets[..., 0]
        phi = phi0.unsqueeze(1) + angular_offsets[..., 1]
        equal_weight = real / multiplicity
        transverse_unit_sum = torch.stack(
            (
                (equal_weight * torch.cos(phi)).sum(dim=1),
                (equal_weight * torch.sin(phi)).sum(dim=1),
            ),
            dim=-1,
        )
        resultant = torch.linalg.vector_norm(transverse_unit_sum, dim=-1).clamp_min(1e-4)
        pt = equal_weight * (jet_pt / resultant).unsqueeze(1)
        energy = pt * torch.cosh(eta)
        particles = eta_phi_pt_e_to_cartesian(
            torch.stack((eta, phi, pt, energy), dim=-1)
        )
        return (particles / float(model_scale)) * real.unsqueeze(-1)

    elif prior_dist == 'axis_aligned_lognormal':
        # Conditioned-axis prior with a random fragmentation hierarchy.  Positive
        # lognormal draws are normalized only over real slots; a common per-jet
        # correction then preserves the requested transverse vector-sum magnitude.
        if jet_features is None or model_scale is None or particle_mask is None:
            raise ValueError(
                "axis_aligned_lognormal requires jet_features, model_scale, and particle_mask"
            )
        if float(model_scale) <= 0:
            raise ValueError("model_scale must be positive")
        device = jet_features.device
        real = particle_mask.to(device=device, dtype=jet_features.dtype)
        if real.shape != (batch_size, num_particles):
            raise ValueError(
                "particle_mask shape must equal (batch_size, num_particles); "
                f"got {tuple(real.shape)}"
            )
        if (real.sum(dim=1) < 1).any():
            raise ValueError(
                "axis_aligned_lognormal requires at least one real particle per jet"
            )

        jet_eta = jet_features[:, 0]
        jet_pt = jet_features[:, 1]
        phi0 = (
            (2 * torch.pi) * torch.rand(batch_size, device=device)
            if jet_phi is None else jet_phi.to(device)
        )
        # Draw angles and log-weights together.  With a fixed seed this makes a jet's
        # sample independent of whether unrelated jets are appended to the batch.
        # Inverse-CDF normals preserve the prefix property of uniform RNG draws:
        # sampling jet 0 alone or as the first row of a larger batch is bit-identical.
        uniform = torch.rand(
            batch_size, num_particles, 3,
            device=device, dtype=jet_features.dtype,
        )
        eps = torch.finfo(jet_features.dtype).eps
        draws = torch.erfinv(2 * uniform.clamp(min=eps, max=1 - eps) - 1) * (2.0**0.5)
        eta = jet_eta.unsqueeze(1) + 0.15 * draws[..., 0]
        phi = phi0.unsqueeze(1) + 0.15 * draws[..., 1]
        positive = torch.exp(draws[..., 2]) * real
        weights = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-12)
        transverse_unit_sum = torch.stack(
            (
                (weights * torch.cos(phi)).sum(dim=1),
                (weights * torch.sin(phi)).sum(dim=1),
            ),
            dim=-1,
        )
        resultant = torch.linalg.vector_norm(
            transverse_unit_sum, dim=-1
        ).clamp_min(1e-4)
        pt = weights * (jet_pt / resultant).unsqueeze(1)
        energy = pt * torch.cosh(eta)
        particles = eta_phi_pt_e_to_cartesian(
            torch.stack((eta, phi, pt, energy), dim=-1)
        )
        return (particles / float(model_scale)) * real.unsqueeze(-1)

    else:
        raise ValueError(f"Unknown prior_dist: {prior_dist}")



def time_dist(batch_size, device='cpu', mode='power_law', **kwargs):
    if mode == 'uniform':
        return torch.rand(batch_size, device=device)
    elif mode == 'power_law':
        # pdf = (1+a) * x^a
        # cdf = x^(a+1)
        # https://math.stackexchange.com/questions/3499892/sampling-from-a-distribution-with-given-pdf
        a = kwargs.get('a', -0.2)
        # sample x uniformly over 0, 1
        u = torch.rand(batch_size, device=device)
        return u ** (1 / (a + 1))
    elif mode == 'lognorm':
        mu = kwargs.get('mu', -0.5)
        sigma = kwargs.get('sigma', 1.0)
        dist = torch.distributions.LogNormal(mu, sigma)
        samples = dist.sample((batch_size,)).to(device)
        # Normalize using the theoretical 95th percentile of this LogNormal so the
        # scaling is batch-independent. p95 = exp(mu + sigma * 1.645).
        p95 = torch.exp(torch.tensor(mu + sigma * 1.645, device=device))
        samples = (samples / p95).clamp(max=1.0)
        return samples
    else:
        raise ValueError(f"Unknown time distribution mode: {mode}")
