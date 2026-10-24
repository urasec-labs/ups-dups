import pytest
from model.context_memory import kv_cache_bytes, estimate
from model.config import ModelConfig

def test_kv_cache_formula():
    assert kv_cache_bytes(2,1,10,2,8,2)==1280
    with pytest.raises(ValueError): kv_cache_bytes(0,1,10,2,8,2)

def test_one_million_token_estimate():
    cfg=ModelConfig(d_model=32,n_heads=4,n_kv_heads=2,n_layers=2,d_ff=64,max_seq_len=1000000)
    result=estimate(cfg,sequence=1000000)
    assert result['kv_cache_bytes']==2*2*1*1000000*2*8*2
