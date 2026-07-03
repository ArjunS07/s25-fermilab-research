import torch


class ModelEMA:
    """Exponential moving average of model parameters (experiment plan 2.1).

    Maintains a shadow copy of the model's state_dict, updated after each optimizer step:
    ``ema = decay*ema + (1-decay)*param``. For flow-matching / diffusion models the EMA
    weights typically give smoother, better generative metrics than the raw weights, so use
    them for evaluation and sampling. Near-free; entirely orthogonal to the architecture.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Detached clones on the same device as the model.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if torch.is_floating_point(v):
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                # Non-float buffers (e.g. integer counters): track exactly.
                s.copy_(v)

    def copy_to(self, model):
        """Load the EMA weights into ``model`` (in place)."""
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, shadow, device=None):
        self.shadow = {k: (v.to(device) if device is not None else v) for k, v in shadow.items()}
