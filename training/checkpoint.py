from pathlib import Path
import torch

def save_checkpoint(path,model,optimizer,step,config_path,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'step':int(step),'config_path':str(config_path),'extra':extra or {}}
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,tmp); tmp.replace(path)

def load_checkpoint(path,model,optimizer=None,map_location='cpu'):
    payload=torch.load(path,map_location=map_location,weights_only=True)
    if not isinstance(payload,dict) or 'model' not in payload or 'step' not in payload: raise ValueError('invalid checkpoint structure')
    model.load_state_dict(payload['model'],strict=True)
    if optimizer is not None and payload.get('optimizer') is not None: optimizer.load_state_dict(payload['optimizer'])
    return payload
