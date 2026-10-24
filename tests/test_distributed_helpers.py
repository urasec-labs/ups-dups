def test_distributed_single_process_cpu():
    from training.distributed import initialize_distributed, cleanup_distributed
    ctx=initialize_distributed()
    assert ctx['world_size']>=1 and ctx['rank']>=0
    cleanup_distributed()
