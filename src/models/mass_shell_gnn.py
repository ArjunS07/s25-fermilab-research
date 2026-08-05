"""Softmax-free Lorentz-equivariant message passing on the regulated mass shell."""
from __future__ import annotations
from typing import NamedTuple, Optional
import torch
import torch.nn as nn
from models.mass_shell_flow import MassShellFlow
from util.mass_shell import log_map, parallel_transport, pushforward_to_tangent
from util.minkowski_utils import dotsq4, normsq4

def _small_normal(layer):
    nn.init.normal_(layer.weight, std=1e-3); nn.init.zeros_(layer.bias)

class Geometry(NamedTuple):
    """Per-forward, layer-invariant context shared by every block (nothing here evolves)."""
    x: torch.Tensor                # (B,N,4) shell positions, float64
    cond: torch.Tensor             # (B,width) conditioning + time embedding
    mask: torch.Tensor             # (B,N) float
    edge: torch.Tensor             # (B,N,N,3) invariant edge features, model dtype
    direction: torch.Tensor        # (B,N,N,4) normalized geodesic directions, float64
    projected_refs: torch.Tensor   # (B,N,2,4) refs pushed to each tangent space, float64
    typed_refs: torch.Tensor       # (B,2,width) role-tagged reference tokens
    support: torch.Tensor          # (B,N,N) bool adjacency (no self, real-real)
    count: torch.Tensor            # (B,N) sqrt(degree), float64
    mass: float

def _geometry(x, mass, dtype):
    rel, d = log_map(x.unsqueeze(2), x.unsqueeze(1), mass, return_distance=True)
    distance = d.squeeze(-1)
    dot = dotsq4(x.unsqueeze(2), x.unsqueeze(1))
    edge = torch.stack((torch.log1p(distance),
                        torch.log1p((dot / mass**2 - 1).clamp_min(0)),
                        distance / (distance + mass)), -1).to(dtype)
    return edge, rel / (mass + distance).unsqueeze(-1)

class MassShellGNNBlock(nn.Module):
    def __init__(self, width, channels, tangent_channels, global_pooling):
        super().__init__()
        self.channels, self.tangent_channels = channels, tangent_channels
        self.global_pooling = global_pooling
        context = width if global_pooling else 0
        self.norm = nn.LayerNorm(width)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * width + 3 + context, width), nn.SiLU(),
                                      nn.Linear(width, width))
        self.node_mlp = nn.Sequential(nn.Linear(3 * width + context, 2 * width), nn.SiLU(),
                                      nn.Linear(2 * width, width))
        if tangent_channels:
            self.direction_gate = nn.Linear(width, channels)
            self.transport_gate = nn.Linear(width, channels)
            self.reference_gate = nn.Sequential(
                nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, channels)
            )
        if global_pooling:
            self.global_mlp = nn.Sequential(nn.LayerNorm(2 * width),
                                            nn.Linear(2 * width, 2 * width), nn.SiLU(),
                                            nn.Linear(2 * width, width))

    def forward(self, h, vectors, global_state, g):
        n = h.shape[1]
        hn = self.norm(h)
        fields = [hn.unsqueeze(2).expand(-1,-1,n,-1),
                  hn.unsqueeze(1).expand(-1,n,-1,-1), g.edge,
                  g.cond[:,None,None,:].expand(-1,n,n,-1)]
        if self.global_pooling:
            fields.append(global_state[:,None,None,:].expand(-1,n,n,-1))
        message = self.edge_mlp(torch.cat(fields,-1)) * g.support.unsqueeze(-1)
        count = g.count
        aggregate = message.sum(2) / count.unsqueeze(-1).to(message.dtype)
        node_fields = [hn, aggregate, g.cond[:,None,:].expand(-1,n,-1)]
        if self.global_pooling:
            node_fields.append(global_state[:,None,:].expand(-1,n,-1))
        h = (h + self.node_mlp(torch.cat(node_fields,-1))) * g.mask.unsqueeze(-1).to(h.dtype)
        if self.tangent_channels:
            direction_gate = (self.direction_gate(message)
                              * g.support.unsqueeze(-1)).to(torch.float64)
            dv = torch.einsum("bijc,bijf->bicf", direction_gate, g.direction)
            sender = vectors.unsqueeze(1).expand(-1,n,-1,-1,-1)
            transported = parallel_transport(g.x.unsqueeze(1).unsqueeze(3),
                                             g.x.unsqueeze(2).unsqueeze(3), sender, g.mass)
            transport_gate = (self.transport_gate(message)
                              * g.support.unsqueeze(-1)).to(torch.float64)
            dv += torch.einsum("bijc,bijcf->bicf", transport_gate, transported)
            ref_input = torch.cat([
                hn.unsqueeze(2).expand(-1, -1, 2, -1),
                g.typed_refs.unsqueeze(1).expand(-1, n, -1, -1),
            ], -1)
            ref_gate = self.reference_gate(ref_input).to(torch.float64)
            dv += torch.einsum("birc,birf->bicf", ref_gate, g.projected_refs)
            vectors = vectors + dv / count.unsqueeze(-1).unsqueeze(-1)
            vectors = pushforward_to_tangent(g.x.unsqueeze(2), vectors, g.mass)
            vectors *= g.mask[:,:,None,None].to(torch.float64)
        if self.global_pooling:
            norm = g.mask.sum(1).clamp_min(1).sqrt().to(h.dtype)
            pooled = (h * g.mask.unsqueeze(-1).to(h.dtype)).sum(1) / norm.unsqueeze(-1)
            global_state = global_state + self.global_mlp(torch.cat([global_state, pooled],-1))
        return h, vectors, global_state

class MassShellGNNBackbone(nn.Module):
    def __init__(self, cond_dim, width, num_layers, vector_channels, regulator_mass,
                 geometric_state, use_global_pooling):
        super().__init__()
        self.mass, self.geometric_state = regulator_mass, geometric_state
        self.tangent = geometric_state == "tangent_channels"
        self.use_global_pooling = use_global_pooling
        self.channels = vector_channels
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim,width),nn.SiLU(),nn.Linear(width,width))
        self.time_embed = nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width))
        # Node seed carries only informative invariants: <x,e_t> (energy) and <x,jet_p4>.
        # <x,x> is identically m^2 on the shell, so it is dropped (a dead constant channel).
        self.node_seed = nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks = nn.ModuleList([MassShellGNNBlock(width,vector_channels,self.tangent,use_global_pooling)
                                     for _ in range(num_layers)])
        self.ref_roles = nn.Parameter(torch.randn(2,width))
        if self.tangent:
            self.channel_readout = nn.Linear(width,vector_channels); _small_normal(self.channel_readout)
        else:
            self.edge_readout = nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,1))
            self.ref_readout = nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))
            _small_normal(self.edge_readout[-1]); _small_normal(self.ref_readout[-1])

    def forward(self, x, t, conditions, mask, references):
        if references is None or references.shape[1] != 2:
            raise ValueError("mass_shell_gnn requires exactly two typed references")
        x64, refs = x.to(torch.float64), references.to(torch.float64)
        dtype = next(self.parameters()).dtype
        tf = torch.stack([t,torch.sin(torch.pi*t),torch.cos(torch.pi*t)],-1)
        cond = self.cond_embed(conditions.to(dtype)) + self.time_embed(tf.to(dtype))
        seed = torch.stack([dotsq4(x64,refs[:,:1]),dotsq4(x64,refs[:,1:2])],-1)
        h = (self.node_seed((seed.sign()*torch.log1p(seed.abs())).to(dtype))+cond[:,None])
        h *= mask.unsqueeze(-1).to(dtype)
        n=x.shape[1]
        support=(mask.bool().unsqueeze(2)&mask.bool().unsqueeze(1)&
                 ~torch.eye(n,device=x.device,dtype=torch.bool).unsqueeze(0))
        count=support.sum(-1).clamp_min(1).to(torch.float64).sqrt()
        edge,direction=_geometry(x64,self.mass,dtype)
        projected=pushforward_to_tangent(x64.unsqueeze(2),refs.unsqueeze(1).expand(-1,n,-1,-1),self.mass)
        ref_norm=torch.sqrt((-normsq4(projected)).clamp_min(0))
        projected=projected/(self.mass+ref_norm).unsqueeze(-1)
        projected = projected * mask[:, :, None, None].to(torch.float64)
        typed_refs = cond[:,None,:] + self.ref_roles[None]
        g = Geometry(x=x64, cond=cond, mask=mask, edge=edge, direction=direction,
                     projected_refs=projected, typed_refs=typed_refs, support=support,
                     count=count, mass=self.mass)
        vectors = (torch.zeros(*x.shape[:2], self.channels, 4, dtype=torch.float64, device=x.device)
                   if self.tangent else None)
        global_state = cond if self.use_global_pooling else None
        for block in self.blocks:
            h,vectors,global_state=block(h,vectors,global_state,g)
        if self.tangent:
            velocity=torch.einsum("bic,bicf->bif",self.channel_readout(h).to(torch.float64),vectors)
        else:
            hi=h.unsqueeze(2).expand(-1,-1,n,-1); hj=h.unsqueeze(1).expand(-1,n,-1,-1)
            coeff=self.edge_readout(torch.cat([hi,hj,edge],-1)).squeeze(-1).to(torch.float64)*support
            velocity=torch.einsum("bij,bijf->bif",coeff,direction)/count.unsqueeze(-1)
            rcoeff=self.ref_readout(torch.cat([h.unsqueeze(2).expand(-1,-1,2,-1),
                typed_refs.unsqueeze(1).expand(-1,n,-1,-1)],-1)).squeeze(-1)
            velocity+=torch.einsum("bir,birf->bif",rcoeff.to(torch.float64),projected)
        return pushforward_to_tangent(x64,velocity,self.mass)*mask.unsqueeze(-1).to(torch.float64)

class MassShellGNNFlow(MassShellFlow):
    def __init__(self,condition_dim,n_particle_types,width,num_layers,vector_channels,
                 regulator_mass,geometric_state,use_global_pooling):
        nn.Module.__init__(self)
        self.cond_dim,self.n_particles_idx=condition_dim,n_particle_types
        self.regulator_mass,self.backbone=regulator_mass,"mass_shell_gnn"
        self.null_cond=nn.Parameter(torch.zeros(condition_dim))
        self.tangent_backbone=MassShellGNNBackbone(condition_dim,width,num_layers,vector_channels,
            regulator_mass,geometric_state,use_global_pooling)
