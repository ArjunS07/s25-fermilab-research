import torch
def mean_std_masked_tensor(var_name, t, mask, pr=True):
        masked_tensor = t * mask.unsqueeze(-1)
        mean = masked_tensor.sum(dim=(0, 1)) / mask.sum()
        std = torch.sqrt(((masked_tensor - mean) ** 2).sum(dim=(0, 1)) / mask.sum())
        if pr:
           print(f"{var_name}.mean: {mean}, std: {std}")
        return mean, std