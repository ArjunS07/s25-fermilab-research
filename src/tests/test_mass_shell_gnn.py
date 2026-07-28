import pytest
import torch
from models.LEFT_JeN import LEFTJeN
from models.mass_shell_gnn import MassShellGNNBlock
from tests.lorentz_test_utils import apply_transform, random_proper_transform
from util.mass_shell import project_to_shell
from util.minkowski_utils import dotsq4
from config import InferRunConfig, infer_config_to_namespace
from util.checkpoint_config import resolve_architecture

def inputs(mass=.1):
    torch.manual_seed(71)
    x=project_to_shell(torch.randn(2,5,4,dtype=torch.float64),mass)
    mask=torch.tensor([[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.float64)
    x[0,4]=torch.tensor([mass,0,0,0],dtype=torch.float64)
    refs=torch.randn(2,2,4,dtype=torch.float64); refs[:,0]=torch.tensor([1.,0,0,0])
    return x,torch.tensor([.2,.8],dtype=torch.float64),torch.randn(2,8),mask,refs

def model(mode,pool=False,mass=.1):
    torch.manual_seed(19)
    return LEFTJeN(5,max_particles=5,hidden_dim=32,num_layers=2,include_pt=True,
        use_reference_vectors=True,backbone="mass_shell_gnn",include_mass_condition=True,
        vector_channels=8,regulator_mass=mass,geometric_state=mode,use_global_pooling=pool).eval()

@pytest.mark.parametrize("mode",["readout_only","tangent_channels"])
@pytest.mark.parametrize("pool",[False,True])
def test_covariance_permutation_padding_tangency(mode,pool):
    x,t,c,mask,refs=inputs(); net=model(mode,pool); out=net(x,t,c,mask,refs)
    L=random_proper_transform(22)
    assert torch.allclose(net(apply_transform(x,L),t,c,mask,apply_transform(refs,L)),
                          apply_transform(out,L),atol=3e-6,rtol=3e-6)
    p=torch.tensor([2,0,3,1,4])
    assert torch.allclose(net(x[:,p],t,c,mask[:,p],refs),out[:,p],atol=3e-6,rtol=3e-6)
    assert torch.equal(out[mask==0],torch.zeros_like(out[mask==0]))
    assert torch.allclose(dotsq4(x,out)*mask,torch.zeros_like(mask),atol=1e-9)

@pytest.mark.parametrize("mass",[.1,.03])
def test_finite_forward_backward_and_backbone_gradients(mass):
    x,t,c,mask,refs=inputs(mass); net=model("readout_only",True,mass).train()
    loss=net(x,t,c,mask,refs).square().sum(); loss.backward()
    assert torch.isfinite(loss)
    assert net.tangent_backbone.blocks[0].edge_mlp[0].weight.grad.abs().sum()>0
    assert net.tangent_backbone.blocks[0].node_mlp[0].weight.grad.abs().sum()>0

def test_signed_sum_is_not_softmax_or_convex_average():
    block=MassShellGNNBlock(4,2,False,False)
    with torch.no_grad(): block.edge_mlp[-1].weight.zero_(); block.edge_mlp[-1].bias.fill_(-2)
    captured={}
    def hook(_,args): captured["x"]=args[0]
    handle=block.node_mlp.register_forward_pre_hook(hook)
    support=~torch.eye(3,dtype=torch.bool).unsqueeze(0)
    block(torch.ones(1,3,4,dtype=torch.float64),torch.zeros(1,3,4),
          torch.zeros(1,3,2,4,dtype=torch.float64),torch.zeros(1,4),torch.zeros(1,4),
          torch.ones(1,3),torch.zeros(1,3,3,3),torch.zeros(1,3,3,4,dtype=torch.float64),
          torch.zeros(1,3,2,4,dtype=torch.float64),torch.zeros(1,2,4),
          support,.1); handle.remove()
    assert torch.allclose(captured["x"][...,4:8],torch.full((1,3,4),-2*2**.5))

def test_padding_exactly_excluded_and_tangent_each_block():
    x,t,c,mask,refs=inputs(); net=model("tangent_channels",True); expected=net(x,t,c,mask,refs)
    changed=x.clone(); changed[0,4]=project_to_shell(torch.tensor([100.,-50.,20.,7.]),.1)
    assert torch.equal(net(changed,t,c,mask,refs)[0,:4],expected[0,:4])
    seen=[]; handles=[b.register_forward_hook(lambda _,__,o: seen.append(o[1].detach()))
                     for b in net.tangent_backbone.blocks]
    net(x,t,c,mask,refs)
    for h in handles:h.remove()
    for vectors in seen:
        error=dotsq4(x.unsqueeze(2),vectors)*mask.unsqueeze(-1)
        assert torch.allclose(error,torch.zeros_like(error),atol=1e-9)

@pytest.mark.parametrize("mode,pool",[("readout_only",False),("tangent_channels",False),
                                       ("readout_only",True),("tangent_channels",True)])
def test_checkpoint_reconstruction(mode,pool):
    a=model(mode,pool)
    model_config = {
        "backbone":"mass_shell_gnn", "geometric_state":mode,
        "use_global_pooling":pool, "n_hidden":32, "n_layers":2,
        "use_reference_vectors":True, "include_mass_condition":True,
        "use_hyperbolic":True, "hyperbolic_model":"mass_shell",
        "regulator_mass":.1, "vector_channels":8,
        "velocity_readout_init":"small_normal",
    }
    args=infer_config_to_namespace(InferRunConfig())
    args,_=resolve_architecture(args,{"full_config":{"model":model_config}})
    b=LEFTJeN(5,max_particles=5,hidden_dim=args.n_hidden,num_layers=args.n_layers,
        include_pt=True,use_reference_vectors=args.use_reference_vectors,
        backbone=args.backbone,include_mass_condition=args.include_mass_condition,
        vector_channels=args.vector_channels,regulator_mass=args.regulator_mass,
        velocity_readout_init=args.velocity_readout_init,geometric_state=args.geometric_state,
        use_global_pooling=args.use_global_pooling)
    b.load_state_dict(a.state_dict())
    x,t,c,mask,refs=inputs(); assert torch.equal(a(x,t,c,mask,refs),b(x,t,c,mask,refs))

def test_tangent_channels_use_typed_reference_directions_and_backpropagate():
    x,t,c,mask,refs=inputs(); net=model("tangent_channels",True).train()
    out=net(x,t,c,mask,refs)
    swapped=net(x,t,c,mask,refs[:,[1,0]])
    assert not torch.allclose(out,swapped,atol=1e-7)
    out.square().sum().backward()
    block=net.tangent_backbone.blocks[0]
    for parameter in (block.edge_mlp[0].weight,block.node_mlp[0].weight,
                      block.direction_gate.weight,block.reference_gate[0].weight,
                      net.tangent_backbone.channel_readout.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum()>0
