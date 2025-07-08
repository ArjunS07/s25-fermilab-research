import torch

# Sourced from LorentzNet paper codebase: https://github.com/sdogsq/LorentzNet-release

def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)

    # 2t^2 - (t^2 + x^2 + y^2 + z^2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    return 2 * psq[..., 0] - psq.sum(dim=-1)
