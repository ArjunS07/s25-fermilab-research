import pickle
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from util.coordinates import transform_rel_particle_coordinates_to_cartesian

class Flow(nn.Module):
    """
    Simple flow model that takes in input tensor and time and outputs transformed tensor.
    Consists of 4 linear layers with ELU activations
    """
    def __init__(self, dim: int = 1, h: int = 64): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, method = 'euler') -> Tensor:
        """
        Calculate the probability density at a particular time step
        """
        if method == 'euler':
            t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
            update = self.forward(x_t=x_t, t=t_start)
            return x_t + update * (t_end - t_start)
            
        elif method == 'RK2':
            # Reshape t_start to be a column vector and expand to match the batch size of x_t
            t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
    
            # Translate x_t by the expected midpoint velocity between t_start and t_end
            start_vel = self.forward(x_t=x_t, t=t_start)
            midpoint_x = x_t + (start_vel * (t_end - t_start) / 2)
            midpoint_vel = self.forward(x_t=midpoint_x, t=t_start + (t_end - t_start) / 2)
    
            return x_t + (t_end - t_start) * midpoint_vel
    
    def clip_gradients(self, max_norm: float = 1.0):
        """
        Clip gradients to prevent exploding gradients
        """
        for param in self.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm)

def simulate_trajectory(flow_model: Flow, x_start: Tensor, t_start: float, t_end: float, num_steps: int = 50) -> Tensor:
    dt = (t_end - t_start) / num_steps
    x_current = x_start.clone().detach()
    
    for i in range(num_steps):
        t_current = t_start + i * dt
        t_next = t_current + dt
        x_current = flow_model.step(x_current, 
                                  torch.tensor(t_current, dtype=torch.float32), 
                                  torch.tensor(t_next, dtype=torch.float32))
    
    return x_current

if __name__ == "__main__":
    with open("data/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("data/x_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    X_train_particle_transformed = transform_rel_particle_coordinates_to_cartesian(X_train).to('cpu')
    
    particle_clouds = X_train_particle_transformed[:, :, :4]
    masks = X_train_particle_transformed[:, :, 4]
    masked_clouds = (masks.unsqueeze(-1) * particle_clouds)
    masked_clouds_flat = masked_clouds.flatten(start_dim=0, end_dim=1)
    non_zero_mask = (masked_clouds_flat != 0).any(dim=1)
    masked_clouds_flat = masked_clouds_flat[non_zero_mask]
    min_std = np.min(np.std(masked_clouds_flat.numpy(), axis=0))
    masked_clouds_flat = masked_clouds_flat / min_std
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = 5000
    ratio = 0.8
    n_train = int(num_samples * ratio)
    x_1 = masked_clouds_flat[torch.randperm(num_samples)]
    x_1 = x_1.to(device)
    flow_model = Flow(dim=4, h=256).to(device)
    x_1_train = x_1[:n_train]
    x_1_val = x_1[n_train:]

    val_time_1 = torch.Tensor([0.1])
    val_time_2 = torch.Tensor([0.9])
    val_times_1 = val_time_1.expand(x_1_val.shape[0], -1).to(device)
    val_times_2 = val_time_2.expand(x_1_val.shape[0], -1).to(device)
    val_times_1.shape, val_times_2.shape
    lr = 1e-2
    weight_decay = 1e-1
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    loss_fn = nn.MSELoss()

    x_0_dist_mean = torch.tensor([0.0])
    x_0_dist_std = torch.tensor([1.5])
    num_epochs = 2000
    losses = []

    val_losses_1 = []
    val_losses_2 = []
    for epoch in range(num_epochs):
        x_0 = (torch.randn(len(x_1_train), 1) * x_0_dist_std + x_0_dist_mean).to(device)
        t = torch.rand(len(x_1_train), 1, device=device)
        x_t = (1 - t) * x_0 + t * x_1_train
        dx_t = x_1_train - x_0
        optimizer.zero_grad()
        loss = loss_fn(flow_model(t=t, x_t=x_t), dx_t)

        x_0_val = torch.randn(len(x_1_val), 1) * x_0_dist_std + x_0_dist_mean

        x_t_val_1 = (1 - val_times_1) * x_0_val + val_times_1 * x_1_val
        dx_t_val_1 = x_1_val - x_0_val
        val_loss_1 = loss_fn(flow_model(t=val_times_1, x_t=x_t_val_1), dx_t_val_1)

        x_t_val_2 = (1 - val_times_2) * x_0_val + val_times_2 * x_1_val
        dx_t_val_2 = x_1_val - x_0_val
        val_loss_2 = loss_fn(flow_model(t=val_times_2, x_t=x_t_val_2), dx_t_val_2)

        val_losses_1.append(val_loss_1.item())
        val_losses_2.append(val_loss_2.item())
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        flow_model.clip_gradients(max_norm=1.0)

        if (epoch) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val loss: t=0.1 {val_loss_1.item():.4f}, t=0.9 {val_loss_2.item():.4f}")

    print("Training complete.")
    sns.lineplot(x=range(len(losses)), y=losses, label="Train loss")
    sns.lineplot(x=range(len(val_losses_1)), y=val_losses_1, label=r"Val loss ($t=0.1$)")
    sns.lineplot(x=range(len(val_losses_2)), y=val_losses_2, label=r"Val loss ($t=0.9$)")
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.savefig("losses.png")

    num_plot_samples = 5000

    initial_gaussian_samples = torch.randn(num_plot_samples, 4) * x_0_dist_std + x_0_dist_mean
    final_samples = simulate_trajectory(flow_model, initial_gaussian_samples, 0, 1.0, num_steps=64)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        sns.histplot(x_1.cpu().numpy()[:, i], alpha=0.7, label='Target', ax=axs[i])
        sns.histplot(final_samples.detach().numpy()[:, i], alpha=0.7, label='Generated', ax=axs[i])
        axs[i].legend()
        axs[i].set_ylabel('Density')
    plt.savefig("feature_distributions.png", )