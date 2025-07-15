import torch
from torch import nn

from util.minkowski_utils import normsq4, dotsq4
    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    return torch.sign(p) * torch.log(torch.abs(p) + 1)

def minkowski_features(x, zero_masks_bool):
    x = x[zero_masks_bool]
    x_i = x.unsqueeze(-2) # second-last dimension - N
    x_j = x.unsqueeze(-3) # third-last dimension - B
    x_diffs = x_i - x_j # (batch_size, n_particles, n_particles, n_features)

    norms = normsq4(x_diffs)
    dots = dotsq4(x_i, x_j)
    norms, dots = psi(norms), psi(dots)
    return norms, dots, x_diffs

N_EDGE_FEATURES = 2
class FMLorentzLayer(nn.Module):
    def __init__(self,n_hidden, 
                 dropout = 0., c_weight=1.0, last_layer=False):
        super(FMLorentzLayer, self).__init__()

        self.c_weight = c_weight

        self.phi_t = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.phi_e = nn.Sequential(
            nn.Linear(N_EDGE_FEATURES, n_hidden, bias=False),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            #  Message + time -> Embedding
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            layer)

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid())
    
    
    def message_passing(self, norms, dots, diffs):
        inp = torch.stack([norms, dots], dim=-1)  # Concatenate along feature dimension
        # print(f"{inp.shape=}")
        out = self.phi_e(inp)
        # print(f"phi_e(norms, dots) = {out.shape}")
        out = out * self.phi_m(out)
        return out


    def forward(self, x_t: torch.Tensor, zero_masks: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi_t = self.phi_t(t.unsqueeze(-1))

        norms, dots, diffs = minkowski_features(x_t, zero_masks)
        messages = self.message_passing(norms, dots, diffs)

        batch_size, n_particles, _, n_hidden = messages.shape
        t_broadcast = phi_t.view(batch_size, 1, 1, -1).expand(-1, n_particles, n_particles, -1)

        # Concatenate messages with time
        messages_with_time = torch.cat([messages, t_broadcast], dim=-1)
        velocity_magnitude = self.phi_x(messages_with_time)
        velocity = velocity_magnitude * diffs
        velocity = torch.mean(velocity, dim=-2)
        
        return velocity
    
from enum import Enum
ode_solver_methods = Enum('ode_solver_methods', ['euler', 'midpoint'])

class LorentzFMNet(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout=0., c_weight=1.0):
        super(LorentzFMNet, self).__init__()
        self.layers = nn.ModuleList([
            FMLorentzLayer(n_hidden, dropout=dropout, c_weight=c_weight)
            for _ in range(n_layers)
        ])

    def forward(self, x_t: torch.Tensor, zero_masks: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        zero_masks_bool = zero_masks.bool()
        for i, layer in enumerate(self.layers):
            if i == 0:
                vel = layer(x_t, zero_masks_bool, t)
            else:
                # Pass the output of the previous layer as input to the next layer
                vel = layer(vel, zero_masks_bool, t)
        return vel

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, method: ode_solver_methods=ode_solver_methods.euler) -> torch.Tensor:
        """
        Calculate the probability density at a particular time step
        """
        if method == ode_solver_methods.euler:
            batch_size = x_t.shape[0]
            t_start = t_start.unsqueeze(0).repeat(batch_size).view(-1, 1, 1)
            t_end = t_end.unsqueeze(0).repeat(batch_size).view(-1, 1, 1)

            # Translate x_t by the expected velocity at t_start
            return x_t + (t_end - t_start) * self.forward(x_t=x_t, t=t_start)
        
        elif method == ode_solver_methods.midpoint:
            batch_size = x_t.shape[0]
            t_start = t_start.unsqueeze(0).repeat(batch_size).view(-1, 1, 1)
            t_end = t_end.unsqueeze(0).repeat(batch_size).view(-1, 1, 1)

            # Translate x_t by the expected midpoint velocity between t_start and t_end
            start_vel = self.forward(x_t=x_t, t=t_start)
            midpoint_x = x_t + (start_vel * (t_end - t_start) / 2)
            midpoint_vel = self.forward(x_t=midpoint_x, t=t_start + (t_end - t_start) / 2)

            return x_t + (t_end - t_start) * midpoint_vel
        else:
            raise ValueError(f"Unknown ODE solver method: {method}")