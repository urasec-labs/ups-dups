import argparse,json
from model.config import load_config

def parameter_counts(c):
    d,v,L,H,KV,ff,E,k=(c.d_model,c.vocab_size,c.n_layers,c.n_heads,c.n_kv_heads,c.d_ff,c.num_experts,c.top_k); head=d//H
    embedding=v*d; attention=d*(H*head)+d*(KV*head)+d*(KV*head)+d*d; norms=4*d; expert=3*d*ff; router=d*E
    moe_layers=sum(1 for i in range(L) if E>1 and i%c.moe_every_n_layers==0); dense_layers=L-moe_layers
    total=embedding+L*(attention+norms)+moe_layers*(E*expert+router)+dense_layers*expert+2*d
    active_expert=moe_layers*k*expert+dense_layers*expert
    active=embedding+L*(attention+norms)+moe_layers*(k*expert+router)+dense_layers*expert+2*d
    return {'total_parameters':int(total),'expert_parameters':int(moe_layers*E*expert),'router_parameters':int(moe_layers*router),'shared_parameters':int(total-moe_layers*E*expert-moe_layers*router),'active_expert_parameters_per_token':int(active_expert),'approx_active_parameters_per_token':int(active),'moe_layers':moe_layers,'dense_layers':dense_layers}

def memory_estimate(c,context,batch=1,bytes_per_value=2):
    kv=2*c.n_layers*context*c.n_kv_heads*(c.d_model//c.n_heads)*batch*bytes_per_value
    scores=batch*c.n_heads*context*context*bytes_per_value
    return {'context_tokens':context,'batch_size':batch,'kv_cache_bytes_approx':kv,'kv_cache_gib_approx':kv/(1024**3),'dense_attention_score_tensor_bytes_if_materialized':scores,'dense_attention_score_tensor_gib_if_materialized':scores/(1024**3),'note':'Excludes weights, activations, optimizer states, and runtime overhead; optimized SDPA may avoid materializing dense score tensor.'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/dev.yaml'); p.add_argument('--context',type=int); p.add_argument('--batch-size',type=int,default=1); a=p.parse_args()
    raw,c=load_config(a.config); counts=parameter_counts(c); total_target=2_800_000_000_000; active_target=104_000_000_000
    print(json.dumps({'config':a.config,'model_config':raw['model'],'parameter_counts':counts,'targets':{'total':total_target,'active':active_target},'target_check':{'total_target_met':counts['total_parameters']>=total_target,'active_target_met':counts['approx_active_parameters_per_token']>=active_target,'total_gap':total_target-counts['total_parameters'],'active_gap':active_target-counts['approx_active_parameters_per_token']},'context_memory':memory_estimate(c,a.context or c.max_seq_len,a.batch_size),'status':'configuration-derived estimate; not a loaded or trained model'},indent=2))
if __name__=='__main__': main()
