import torch
from torch import nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        if head_dim % 2: raise ValueError('head_dim must be even')
        self.register_buffer('inv_freq',1.0/(theta**(torch.arange(0,head_dim,2).float()/head_dim)),persistent=False)
    def forward(self,x,offset=0):
        pos=torch.arange(offset,offset+x.shape[-2],device=x.device,dtype=self.inv_freq.dtype)
        freqs=torch.outer(pos,self.inv_freq.to(x.device)); emb=torch.cat((freqs,freqs),dim=-1)
        return emb.cos().to(x.dtype)[None,None],emb.sin().to(x.dtype)[None,None]

def rotate_half(x):
    a,b=x.chunk(2,dim=-1); return torch.cat((-b,a),dim=-1)

def apply_rope(q,k,cos,sin): return q*cos+rotate_half(q)*sin,k*cos+rotate_half(k)*sin

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,n_kv_heads,dropout,rope_theta):
        super().__init__(); self.n_heads=n_heads; self.n_kv_heads=n_kv_heads; self.head_dim=d_model//n_heads; self.dropout=dropout
        self.q_proj=nn.Linear(d_model,n_heads*self.head_dim,bias=False); self.k_proj=nn.Linear(d_model,n_kv_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(d_model,n_kv_heads*self.head_dim,bias=False); self.o_proj=nn.Linear(d_model,d_model,bias=False)
        self.rope=RotaryEmbedding(self.head_dim,rope_theta)
    def forward(self,x,past_key_value=None,use_cache=False):
        b,t,c=x.shape; past_len=0
        q=self.q_proj(x).view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(x).view(b,t,self.n_kv_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(x).view(b,t,self.n_kv_heads,self.head_dim).transpose(1,2)
        if past_key_value is not None: past_len=past_key_value[0].shape[-2]
        cos,sin=self.rope(q,offset=past_len); q,k=apply_rope(q,k,cos,sin)
        if past_key_value is not None:
            k=torch.cat((past_key_value[0],k),dim=-2); v=torch.cat((past_key_value[1],v),dim=-2)
        present=(k,v) if use_cache else None
        if self.n_kv_heads != self.n_heads:
            rep=self.n_heads//self.n_kv_heads; k_attn=k.repeat_interleave(rep,dim=1); v_attn=v.repeat_interleave(rep,dim=1)
        else: k_attn,v_attn=k,v
        # Cached multi-token chunks need an offset causal mask; SDPA's non-square is_causal
        # alignment is upper-left and would otherwise expose the wrong positions.
        if past_len > 0 and t > 1:
            q_pos=torch.arange(past_len,past_len+t,device=x.device)[:,None]
            k_pos=torch.arange(past_len+t,device=x.device)[None,:]
            mask=k_pos <= q_pos
            y=F.scaled_dot_product_attention(q,k_attn,v_attn,attn_mask=mask,dropout_p=self.dropout if self.training else 0.0,is_causal=False)
        else:
            y=F.scaled_dot_product_attention(q,k_attn,v_attn,dropout_p=self.dropout if self.training else 0.0,is_causal=(past_len == 0 and t > 1))
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,c)),present
