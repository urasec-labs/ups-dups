import torch
from training.checkpoint import save_checkpoint,load_checkpoint

def test_checkpoint_roundtrip(tmp_path,tiny_model):
 opt=torch.optim.AdamW(tiny_model.parameters()); path=tmp_path/'model.pt'; save_checkpoint(path,tiny_model,opt,7,'configs/dev.yaml',{'tag':'test'})
 other=type(tiny_model)(tiny_model.config); payload=load_checkpoint(path,other); assert payload['step']==7 and payload['extra']['tag']=='test'
 for a,b in zip(tiny_model.parameters(),other.parameters()): assert torch.equal(a,b)
