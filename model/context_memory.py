"""Transparent KV-cache memory estimates; estimates are not runtime benchmarks."""
def kv_cache_bytes(layers, batch, sequence, kv_heads, head_dim, bytes_per_element=2):
    values=(layers,batch,sequence,kv_heads,head_dim,bytes_per_element)
    if any(not isinstance(v,int) or v <= 0 for v in values): raise ValueError('all arguments must be positive integers')
    return 2*layers*batch*sequence*kv_heads*head_dim*bytes_per_element

def human_bytes(n):
    units=['B','KiB','MiB','GiB','TiB','PiB']
    x=float(n)
    for unit in units:
        if x < 1024 or unit==units[-1]: return f'{x:.2f} {unit}'
        x/=1024

def estimate(config, sequence=1_000_000, batch=1, bytes_per_element=2):
    head_dim=config.d_model//config.n_heads
    cache=kv_cache_bytes(config.n_layers,batch,sequence,config.n_kv_heads,head_dim,bytes_per_element)
    # Dense attention score matrix per layer/head is an upper-bound illustration, not necessarily materialized by SDPA.
    score_bytes=batch*config.n_heads*sequence*sequence*4
    return {'sequence_tokens':sequence,'batch_size':batch,'kv_cache_bytes':cache,'kv_cache_human':human_bytes(cache),'dense_attention_scores_bytes_per_layer_fp32':score_bytes,'dense_attention_scores_human_per_layer_fp32':human_bytes(score_bytes),'note':'KV cache only; excludes weights, activations, allocator overhead, temporary buffers, and training activations.'}
