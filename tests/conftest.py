import pytest
from model.config import ModelConfig
from model.transformer import UpsDupsLM
@pytest.fixture
def tiny_config(): return ModelConfig(vocab_size=260,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,d_ff=64,max_seq_len=32,dropout=0.,num_experts=4,top_k=2)
@pytest.fixture
def tiny_model(tiny_config): return UpsDupsLM(tiny_config)
