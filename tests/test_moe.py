import torch
from model.moe import TopKMoE

def test_moe_output_aux_and_backward():
 m=TopKMoE(16,32,4,2); x=torch.randn(2,5,16,requires_grad=True); y,aux=m(x)
 assert y.shape==x.shape and aux.ndim==0 and torch.isfinite(aux); (y.square().mean()+aux).backward(); assert m.router.weight.grad is not None
 assert int(m.last_load.sum())==2*5*2

def test_moe_router_loads():
 m=TopKMoE(8,16,3,1); m(torch.randn(1,4,8)); assert m.last_load.shape==(3,)
