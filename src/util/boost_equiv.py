import torch
import torch.nn.functional as F


"""Primitive util
"""

def __get_boost_mat(beta_vec):
    beta_x, beta_y, beta_z = beta_vec[:, 0], beta_vec[:, 1], beta_vec[:, 2]
    beta_squared = torch.sum(beta_vec**2, dim=-1, keepdim=True)  # [batch, 1, 1]
    
    beta_squared = torch.sum(beta_vec**2, dim=-1)  # [batch]
    gamma = 1.0 / torch.sqrt(torch.clamp(1 - beta_squared, min=1e-8))  # [batch]
    gamma_minus_1 = gamma - 1.0
    
    # Row 0: [γ, -γβₓ, -γβᵧ, -γβᵤ]
    row_0 = torch.stack([
        gamma,
        -gamma * beta_x,
        -gamma * beta_y, 
        -gamma * beta_z
    ], dim=1)  # [batch, 4]
    
    # Row 1: [-γβₓ, 1+(γ-1)βₓ²/β², (γ-1)βₓβᵧ/β², (γ-1)βₓβᵤ/β²]
    beta_sq_safe = torch.clamp(beta_squared, min=1e-8)
    factor = gamma_minus_1 / beta_sq_safe
    
    row_1 = torch.stack([
        -gamma * beta_x,
        1.0 + factor * beta_x**2,
        factor * beta_x * beta_y,
        factor * beta_x * beta_z
    ], dim=1)
    
    # Row 2: [-γβᵧ, (γ-1)βᵧβₓ/β², 1+(γ-1)βᵧ²/β², (γ-1)βᵧβᵤ/β²]
    row_2 = torch.stack([
        -gamma * beta_y,
        factor * beta_y * beta_x,
        1.0 + factor * beta_y**2,
        factor * beta_y * beta_z
    ], dim=1)
    
    # Row 3: [-γβᵤ, (γ-1)βᵤβₓ/β², (γ-1)βᵤβᵧ/β², 1+(γ-1)βᵤ²/β²]
    row_3 = torch.stack([
        -gamma * beta_z,
        factor * beta_z * beta_x,
        factor * beta_z * beta_y,
        1.0 + factor * beta_z**2
    ], dim=1)
    
    return torch.stack([row_0, row_1, row_2, row_3], dim=1)  # [batch, 4, 4]


def __compute_total_4momentum(p_4vec, mask):
    mask_expanded = mask.unsqueeze(-1)  # [batch, 150, 1]
    masked_p_4vec = p_4vec * mask_expanded
    total_4momentum = torch.sum(masked_p_4vec, dim=1)  # [batch, 4]
    return total_4momentum

def __lorentz_boost(p_4vec, beta_vec, mask):
    """
    Boost a batch of particle clouds with the given beta value
    """
    boost_matrix = __get_boost_mat(beta_vec)
    boost_matrix_expanded = boost_matrix.unsqueeze(1)  # [batch, 1, 4, 4]

    p_4vec_expanded = p_4vec.unsqueeze(-1)  # [batch, 150, 4, 1]
    boosted_p_4vec = torch.matmul(boost_matrix_expanded, p_4vec_expanded)
    boosted_p_4vec = boosted_p_4vec.squeeze(-1)  # [batch, 150, 4]
    
    mask_expanded = mask.unsqueeze(-1)
    boosted_p_4vec = boosted_p_4vec * mask_expanded + p_4vec * (1 - mask_expanded)

    return boosted_p_4vec

def boost_to_com_frame(p_4vec, mask):
    total_4momentum = __compute_total_4momentum(p_4vec, mask)
    
    total_E = total_4momentum[:, 0:1]  # [batch, 1]
    total_p_3vec = total_4momentum[:, 1:4]  # [batch, 3]
    beta_vec = -total_p_3vec / torch.clamp(total_E, min=1e-8)  # [batch, 3]
    
    p_4vec_com = __lorentz_boost(p_4vec, beta_vec, mask)
    
    return p_4vec_com

def enforce_com_frame(p_4vec, mask):
    """
    Roughly accurate correction to translate momenta to 0 total
    """
    E = p_4vec[..., 0:1]  # [batch, 150, 1]
    p_3vec = p_4vec[..., 1:4]  # [batch, 150, 3]
    
    mask_expanded = mask.unsqueeze(-1)  # [batch, 150, 1]
    masked_p_3vec = p_3vec * mask_expanded
    total_p_3vec = torch.sum(masked_p_3vec, dim=1, keepdim=True)  # [batch, 1, 3]
    
    N_valid = torch.clamp(torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1), min=1)
    
    # Redistribute momentum equally among valid particles to zero total
    momentum_correction = total_p_3vec / N_valid  # [batch, 1, 3]
    p_3vec_corrected = p_3vec - momentum_correction * mask_expanded
    
    # Recalculate energies for massless particles: E = |p|
    E_corrected = torch.norm(p_3vec_corrected, dim=-1, keepdim=True)
    
    # Only apply correction to valid particles
    E_final = E_corrected * mask_expanded + E * (1 - mask_expanded)
    p_3vec_final = p_3vec_corrected * mask_expanded + p_3vec * (1 - mask_expanded)
    
    p_4vec_corrected = torch.cat([E_final, p_3vec_final], dim=-1)
    
    return p_4vec_corrected


def boost_from_com_to_lab(p_4vec_com, target_total_4momentum, mask):
    # Current total in CoM frame (should be ~(M, 0, 0, 0))
    current_total = __compute_total_4momentum(p_4vec_com, mask)
    
    # Extract target quantities
    target_E = target_total_4momentum[:, 0:1]  # [batch, 1]
    target_p_3vec = target_total_4momentum[:, 1:4]  # [batch, 3]
    
    # Boost velocity to achieve target
    beta_vec = target_p_3vec / torch.clamp(target_E, min=1e-8)  # [batch, 3]
    
    # Apply boost
    p_4vec_lab = __lorentz_boost(p_4vec_com, beta_vec, mask)
    
    return p_4vec_lab