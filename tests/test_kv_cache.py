import torch, pytest

def test_cached_logits_match_full_prefix(tiny_model):
    tiny_model.eval(); ids=torch.randint(0,260,(1,9))
    with torch.no_grad():
        full=tiny_model(ids)['logits'][:,-1,:]
        pre=tiny_model(ids[:,:-1],use_cache=True)
        cached=tiny_model(ids[:,-1:],past_key_values=pre['past_key_values'],use_cache=True)['logits'][:,-1,:]
    assert torch.allclose(full,cached,atol=1e-5,rtol=1e-4)

def test_generation_uses_context_limit(tiny_model):
    ids=torch.randint(0,260,(1,30))
    with pytest.raises(ValueError,match='prompt plus max_new_tokens'):
        tiny_model.generate(ids,max_new_tokens=3)

def test_generation_kv_cache(tiny_model):
    ids=torch.randint(0,260,(1,5))
    out=tiny_model.generate(ids,max_new_tokens=3,top_k=1)
    assert out.shape==(1,8)

def test_cached_multi_token_chunk_matches_full_prefix(tiny_model):
    tiny_model.eval(); ids=torch.randint(0,260,(1,9))
    with torch.no_grad():
        full=tiny_model(ids)['logits'][:,-3:,:]
        pre=tiny_model(ids[:,:-3],use_cache=True)
        cached=tiny_model(ids[:,-3:],past_key_values=pre['past_key_values'],use_cache=True)['logits']
    assert torch.allclose(full,cached,atol=1e-5,rtol=1e-4)
