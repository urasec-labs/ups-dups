from dataclasses import dataclass
from typing import Any
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 128
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    dropout: float = 0.0
    num_experts: int = 4
    top_k: int = 2
    moe_every_n_layers: int = 1
    expert_capacity_factor: float = 1.25

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        obj = cls(**data); obj.validate(); return obj

    def validate(self):
        if self.vocab_size < 260: raise ValueError("vocab_size must be >= 260")
        if min(self.d_model,self.n_layers,self.d_ff,self.n_heads,self.n_kv_heads) <= 0: raise ValueError("dimensions must be positive")
        if self.n_heads % self.n_kv_heads or self.d_model % self.n_heads: raise ValueError("head counts/dimensions are incompatible")
        if (self.d_model // self.n_heads) % 2: raise ValueError("head dimension must be even for RoPE")
        if self.max_seq_len < 2: raise ValueError("max_seq_len must be >= 2")
        if not 1 <= self.top_k <= self.num_experts: raise ValueError("require 1 <= top_k <= num_experts")
        if self.moe_every_n_layers < 1: raise ValueError("moe_every_n_layers must be >= 1")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0, 1)")

def load_config(path: str):
    with open(path, encoding='utf-8') as f: raw=yaml.safe_load(f)
    if not isinstance(raw,dict) or 'model' not in raw: raise ValueError('config must contain model mapping')
    return raw, ModelConfig.from_dict(raw['model'])
