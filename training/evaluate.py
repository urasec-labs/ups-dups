import torch
@torch.no_grad()
def evaluate(model,loader,device='cpu',max_batches=20):
    was=model.training; model.eval(); losses=[]
    for i,(x,y) in enumerate(loader):
        if i>=max_batches: break
        out=model(x.to(device),y.to(device))
        if out['loss'] is not None: losses.append(float(out['loss'].item()))
    model.train(was)
    return {'batches':len(losses),'mean_loss':sum(losses)/len(losses) if losses else None}
