import torch
from torch import nn
import torch.nn.functional as F
from model.attention import CausalSelfAttention
from model.moe import TopKMoE

class DenseFFN(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__(); self.gate=nn.Linear(d_model,d_ff,bias=False); self.up=nn.Linear(d_model,d_ff,bias=False); self.down=nn.Linear(d_ff,d_model,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class DecoderBlock(nn.Module):
    def __init__(self,cfg,index):
        super().__init__(); self.norm1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg.d_model,cfg.n_heads,cfg.n_kv_heads,cfg.dropout,cfg.rope_theta); self.norm2=nn.LayerNorm(cfg.d_model)
        self.is_moe=cfg.num_experts>1 and index%cfg.moe_every_n_layers==0
        self.ffn=TopKMoE(cfg.d_model,cfg.d_ff,cfg.num_experts,cfg.top_k,cfg.expert_capacity_factor) if self.is_moe else DenseFFN(cfg.d_model,cfg.d_ff)
    def forward(self,x,past_key_value=None,use_cache=False):
        attn_out,present=self.attn(self.norm1(x),past_key_value=past_key_value,use_cache=use_cache)
        x=x+attn_out; aux=x.new_zeros(())
        if self.is_moe: y,aux=self.ffn(self.norm2(x))
        else: y=self.ffn(self.norm2(x))
        return x+y,aux,present

class UpsDupsLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); cfg.validate(); self.config=cfg
        self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.d_model); self.dropout=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([DecoderBlock(cfg,i) for i in range(cfg.n_layers)]); self.final_norm=nn.LayerNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.lm_head.weight=self.token_embedding.weight; self.aux_loss_coef=.01
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,mean=0.,std=.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,mean=0.,std=.02)
    def forward(self,input_ids,targets=None,past_key_values=None,use_cache=False):
        if input_ids.ndim!=2: raise ValueError('input_ids must have shape [batch, sequence]')
        if past_key_values is not None and len(past_key_values) != len(self.blocks): raise ValueError('past_key_values must contain one cache per decoder block')
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_len + input_ids.shape[1] > self.config.max_seq_len: raise ValueError(f'total sequence length {past_len + input_ids.shape[1]} exceeds configured max_seq_len {self.config.max_seq_len}')
        x=self.dropout(self.token_embedding(input_ids)); aux=x.new_zeros(()); presents=[]
        for i,block in enumerate(self.blocks):
            past=None if past_key_values is None else past_key_values[i]
            x,a,present=block(x,past_key_value=past,use_cache=use_cache); aux=aux+a
            if use_cache: presents.append(present)
        logits=self.lm_head(self.final_norm(x)); loss=None
        if targets is not None:
            if targets.shape!=input_ids.shape: raise ValueError('targets must have same shape as input_ids')
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) + self.aux_loss_coef*aux
        return {'logits':logits,'loss':loss,'aux_loss':aux,'past_key_values':tuple(presents) if use_cache else None}
    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens=50,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.):
        from inference.sampling import sample_next_token
        if input_ids.ndim!=2 or input_ids.shape[0]<1: raise ValueError('input_ids must be non-empty [batch, sequence]')
        if max_new_tokens<0: raise ValueError('max_new_tokens must be non-negative')
        if input_ids.shape[1] < 1: raise ValueError('input_ids sequence must not be empty')
        if input_ids.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8): raise TypeError('input_ids must contain integer token IDs')
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size): raise ValueError('input_ids contain token IDs outside configured vocabulary')
        if input_ids.shape[1] > self.config.max_seq_len: raise ValueError('prompt exceeds configured max_seq_len')
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len: raise ValueError('prompt plus max_new_tokens exceeds configured max_seq_len; increase config or reduce generation length')
        was_training=self.training; self.eval(); out=input_ids; cache=None
        try:
            result=self(input_ids,use_cache=True); logits=result['logits'][:,-1,:]; cache=result['past_key_values']
            for step in range(max_new_tokens):
                nxt=sample_next_token(logits,out,temperature,top_k,top_p,repetition_penalty)
                out=torch.cat([out,nxt],dim=1)
                if step + 1 < max_new_tokens:
                    result=self(nxt,past_key_values=cache,use_cache=True)
                    logits=result['logits'][:,-1,:]; cache=result['past_key_values']
            return out
        finally:
            self.train(was_training)
