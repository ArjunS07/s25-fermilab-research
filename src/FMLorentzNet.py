import torch
from torch import nn

from util.minkowski_utils import normsq4, dotsq4
from enum import Enum
ode_solver_methods = Enum('ode_solver_methods', ['euler', 'midpoint'])

    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    # Clamp inputs
    p = torch.clamp(p, -1e6, 1e6)
    return torch.sign(p) * torch.log1p(torch.abs(p))

def minkowski_features(x, device='cpu'):
    # print(f"{x.shape=}")
    x_i = x.unsqueeze(-2).to(device)  # second-last dimension - N
    x_j = x.unsqueeze(-3).to(device)  # third-last dimension - B
    x_diffs = x_i - x_j  # (batch_size, n_particles, n_particles, 

    norms = normsq4(x_diffs).to(device)
    dots = dotsq4(x_i, x_j).to(device)
    norms, dots = psi(norms), psi(dots)
    return norms, dots, x_diffs

N_EDGE_FEATURES = 2
class FMLorentzLayer(nn.Module):
    def __init__(self,n_hidden, 
                 dropout = 0., c_weight=1.0, last_layer=False, device='cpu'):
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

        self.phi_x = nn.Sequential(
            nn.LayerNorm(n_hidden * 2),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 4, bias=True))

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1))
    
        self.device = device

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
    
    
    def message_passing(self, norms, dots, diffs):
        # inp = torch.stack([norms, dots], dim=-1)  # Concatenate along feature dimension
        inp = torch.cat([norms.unsqueeze(-1), dots.unsqueeze(-1)], dim=-1).to(self.device)
        # print(f"{inp.shape=} {inp.mean()=}, {inp.std()=}")
        base = self.phi_e(inp).to(self.device)
        scale = torch.tanh(self.phi_m(base))  # restrict to [-1, 1]
        out = base * (1 + scale)
        return out.to(self.device)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi_t = self.phi_t(t.unsqueeze(-1)).to(self.device)
        # print(f"{phi_t.shape=}")

        norms, dots, diffs = minkowski_features(x_t, self.device)
        # print(f"{norms.shape=}, {dots.shape=}, {diffs.shape=}")
        messages = self.message_passing(norms, dots, diffs).to(self.device)
        # print(f"{messages.shape=}, {phi_t.shape=}")

        batch_size, n_particles, _, n_hidden = messages.shape
        t_broadcast = phi_t.view(batch_size, 1, 1, -1).expand(-1, n_particles, n_particles, -1).to(self.device)

        # Concatenate messages with time
        messages_with_time = torch.cat([messages, t_broadcast], dim=-1).to(self.device)
        velocity_magnitude = self.phi_x(messages_with_time).to(self.device)
        diffs = diffs / (torch.norm(diffs, dim=-1, keepdim=True) + 1e-6)
        velocity = (velocity_magnitude * diffs).to(self.device)
        velocity = torch.sum(velocity, dim=-2).to(self.device)
        return velocity
    
class LorentzFMNet(nn.Module):
    def __init__(self, n_hidden, n_layers, dropout=0., c_weight=1.0, device='cpu'):
        super(LorentzFMNet, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([
            FMLorentzLayer(n_hidden, dropout=dropout, c_weight=c_weight, device=device)
            for _ in range(n_layers)
        ])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i == 0:
                vel = layer(x_t, t).to(self.device)
                print(f"Layer {i}: vel shape={vel.shape}, mean={vel.mean()}, std={vel.std()}")
            else:
                vel = layer(vel, t)
                print(f"Layer {i}: vel shape={vel.shape}, mean={vel.mean()}, std={vel.std()}")
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
            update = self.forward(x_t=x_t, t=t_start)
            # print(f"Update: mean={update.mean()}, std={update.std()}")
            return x_t + (t_end - t_start) * update

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