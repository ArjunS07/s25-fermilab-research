import torch

# Sourced from LorentzNet paper codebase: https://github.com/sdogsq/LorentzNet-release

def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)

    # 2t^2 - (t^2 + x^2 + y^2 + z^2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def spacelike_mask(p, tolerance_factor=64.0):
    """Classify materially spacelike vectors with a roundoff-aware threshold.

    Computing ``E²-|p|²`` is cancellation-prone near the light cone.  The
    tolerance scales with the magnitudes of both cancelling terms and the input
    dtype epsilon; it does not excuse physically meaningful negative mass².
    """
    psq = p.square()
    energy_sq = psq[..., 0]
    momentum_sq = psq[..., 1:].sum(dim=-1)
    msq = energy_sq - momentum_sq
    scale = (energy_sq + momentum_sq).clamp_min(torch.finfo(p.dtype).tiny)
    tolerance = tolerance_factor * torch.finfo(p.dtype).eps * scale
    return msq < -tolerance


def relative_mass_shell_residual(p):
    """Return ``|E²-|p|²| / (E²+|p|²)`` without a zero-vector NaN."""
    psq = p.square()
    energy_sq = psq[..., 0]
    momentum_sq = psq[..., 1:].sum(dim=-1)
    scale = (energy_sq + momentum_sq).clamp_min(torch.finfo(p.dtype).tiny)
    return (energy_sq - momentum_sq).abs() / scale
    
def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    return 2 * psq[..., 0] - psq.sum(dim=-1)
