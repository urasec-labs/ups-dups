"""Minimal torchrun/DDP helpers; multi-node execution has not been validated here."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def initialize_distributed():
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    if world_size > 1 and not dist.is_initialized():
        backend='nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
    rank=dist.get_rank() if dist.is_initialized() else 0
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank); device=torch.device('cuda',local_rank)
    else: device=torch.device('cpu')
    return {'distributed':dist.is_initialized(),'rank':rank,'local_rank':local_rank,'world_size':dist.get_world_size() if dist.is_initialized() else 1,'device':device}

def wrap_ddp(model, context):
    if not context['distributed']: return model
    kwargs={'device_ids':[context['local_rank']]} if context['device'].type=='cuda' else {}
    return DistributedDataParallel(model, **kwargs)

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()
