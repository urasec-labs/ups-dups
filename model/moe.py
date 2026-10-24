import torch
from torch import nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__(); self.gate=nn.Linear(d_model,d_ff,bias=False); self.up=nn.Linear(d_model,d_ff,bias=False); self.down=nn.Linear(d_ff,d_model,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class TopKMoE(nn.Module):
    """Reference local-process MoE; not distributed expert parallelism."""
    def __init__(self,d_model,d_ff,num_experts,top_k,capacity_factor=1.25):
        super().__init__()
        if not 1 <= top_k <= num_experts: raise ValueError('invalid top_k')
        self.num_experts=num_experts; self.top_k=top_k; self.capacity_factor=capacity_factor
        self.router=nn.Linear(d_model,num_experts,bias=False); self.experts=nn.ModuleList([Expert(d_model,d_ff) for _ in range(num_experts)]); self.last_load=None
    def forward(self,x):
        shape=x.shape; flat=x.reshape(-1,shape[-1]); logits=self.router(flat); probs=logits.softmax(-1)
        weights,indices=probs.topk(self.top_k,dim=-1); weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
        out=torch.zeros_like(flat); loads=torch.zeros(self.num_experts,device=x.device,dtype=torch.long)
        for e,expert in enumerate(self.experts):
            token_idx,slot_idx=torch.where(indices==e); loads[e]=token_idx.numel()
            if token_idx.numel(): out.index_add_(0,token_idx,expert(flat.index_select(0,token_idx))*weights[token_idx,slot_idx,None])
        self.last_load=loads.detach()
        assignment=torch.nn.functional.one_hot(indices,num_classes=self.num_experts).float().mean(dim=(0,1))
        aux=self.num_experts*torch.sum(assignment*probs.mean(dim=0))
        return out.view(shape),aux
