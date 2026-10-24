from model.parameter_report import parameter_counts,memory_estimate

def test_parameter_count_consistency(tiny_config):
 r=parameter_counts(tiny_config); assert r['total_parameters']>r['expert_parameters']>0; assert r['router_parameters']>0; assert r['approx_active_parameters_per_token']<=r['total_parameters']
def test_memory_estimate_scales(tiny_config):
 a=memory_estimate(tiny_config,128); b=memory_estimate(tiny_config,256); assert b['kv_cache_bytes_approx']==2*a['kv_cache_bytes_approx']
