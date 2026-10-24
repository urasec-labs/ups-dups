"""Small-scale causal LM trainer. CPU smoke training is supported; production distributed training is not yet integrated."""
import argparse, math, random, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model.config import load_config
from model.transformer import UpsDupsLM
from model.tokenizer import ByteBPETokenizer
from training.dataset import TokenBlockDataset, StreamingTokenBlockDataset
from training.checkpoint import save_checkpoint, load_checkpoint
from training.evaluate import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/dev.yaml'); p.add_argument('--data',default=None); p.add_argument('--steps',type=int,default=None)
    p.add_argument('--resume',default=None); p.add_argument('--validation-data',default=None)
    p.add_argument('--streaming',action='store_true',help='Stream text file(s); requires an independent --validation-data path')
    p.add_argument('--tokenizer',default=None,help='Trained BPE tokenizer JSON; configured model vocab_size must match it')
    p.add_argument('--amp',action='store_true',help='Use CUDA autocast (BF16 if supported, otherwise FP16)')
    a=p.parse_args()
    raw,cfg=load_config(a.config); tr=raw.get('training',{});
    if a.tokenizer and ByteBPETokenizer.load(a.tokenizer).vocab_size != cfg.vocab_size: raise SystemExit(f'tokenizer vocab size does not match model config: tokenizer={ByteBPETokenizer.load(a.tokenizer).vocab_size}, config={cfg.vocab_size}')
    seed=int(raw.get('seed',1337)); random.seed(seed); torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.amp and device.type!='cuda': raise SystemExit('--amp requires CUDA; no silent CPU fallback')
    model=UpsDupsLM(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(tr.get('learning_rate',3e-4)),weight_decay=float(tr.get('weight_decay',.01))); step0=0
    if a.resume: step0=load_checkpoint(a.resume,model,opt,map_location=device)['step']
    data_path=Path(a.data or 'data/smoke_train.txt'); seq=min(cfg.max_seq_len,int(tr.get('seq_len',cfg.max_seq_len))); batch=int(tr.get('batch_size',2))
    if a.streaming:
        if not a.validation_data: raise SystemExit('--streaming requires --validation-data to prevent accidental train/validation leakage')
        if data_path.resolve() == Path(a.validation_data).resolve(): raise SystemExit('training and validation paths must differ')
        train_paths=sorted(data_path.glob('*.txt')) if data_path.is_dir() else [data_path]
        val_path=Path(a.validation_data); val_paths=sorted(val_path.glob('*.txt')) if val_path.is_dir() else [val_path]
        if not train_paths or not val_paths or any(not q.is_file() for q in train_paths+val_paths): raise SystemExit('Training/validation shard paths must exist and contain .txt files')
        train_ds=StreamingTokenBlockDataset(train_paths,seq,tokenizer_path=a.tokenizer); val_ds=StreamingTokenBlockDataset(val_paths,seq,tokenizer_path=a.tokenizer)
        train_dl=DataLoader(train_ds,batch_size=batch,drop_last=True); val_dl=DataLoader(val_ds,batch_size=batch)
    else:
        if a.validation_data and data_path.resolve() == Path(a.validation_data).resolve(): raise SystemExit('training and validation paths must differ')
        if a.validation_data:
            train_ds=TokenBlockDataset(data_path,seq,'all',seed=seed,tokenizer_path=a.tokenizer); val_ds=TokenBlockDataset(a.validation_data,seq,'all',seed=seed,tokenizer_path=a.tokenizer)
        else:
            train_ds=TokenBlockDataset(data_path,seq,'train',seed=seed,tokenizer_path=a.tokenizer); val_ds=TokenBlockDataset(data_path,seq,'val',seed=seed,tokenizer_path=a.tokenizer)
        train_dl=DataLoader(train_ds,batch_size=batch,shuffle=True,drop_last=True); val_dl=DataLoader(val_ds,batch_size=batch,shuffle=False)
        if len(train_dl)==0: raise SystemExit('Dataset too small for selected batch/sequence configuration.')
    iterator=iter(train_dl); max_steps=a.steps or int(tr.get('max_steps',20)); accum=int(tr.get('grad_accum_steps',1)); warmup=int(tr.get('warmup_steps',0)); base=float(tr.get('learning_rate',3e-4)); min_lr=float(tr.get('min_learning_rate',base)); ckpt=Path(tr.get('checkpoint_dir','checkpoints')); ckpt.mkdir(parents=True,exist_ok=True)
    scaler=torch.amp.GradScaler('cuda',enabled=(a.amp and not torch.cuda.is_bf16_supported()))
    print('device:',device,'amp:',a.amp,'validation_before_training:',evaluate(model,val_dl,device))
    started=time.time()
    for step in range(step0+1,step0+max_steps+1):
        lr=base*(step/max(1,warmup)) if warmup and step<=warmup else min_lr+(base-min_lr)*.5*(1+math.cos(math.pi*min(1,max(0,(step-warmup)/max(1,max_steps-warmup)))))
        for group in opt.param_groups: group['lr']=lr
        opt.zero_grad(set_to_none=True); loss_value=0.
        for _ in range(accum):
            try: x,y=next(iterator)
            except StopIteration:
                iterator=iter(train_dl)
                try: x,y=next(iterator)
                except StopIteration: raise SystemExit('Training stream did not produce a full batch; reduce batch/sequence length or add more data.')
            with torch.autocast(device_type=device.type, dtype=(torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16), enabled=a.amp):
                loss=model(x.to(device),y.to(device))['loss']/accum
            scaler.scale(loss).backward(); loss_value+=float(loss.detach())
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(tr.get('grad_clip',1.))); scaler.step(opt); scaler.update()
        line=f'step={step} loss={loss_value:.6f} lr={lr:.8g} elapsed_s={time.time()-started:.1f}'; print(line)
        with (ckpt/'training.log').open('a',encoding='utf-8') as f: f.write(line+'\n')
        if step%int(tr.get('eval_interval',10))==0 or step==step0+max_steps:
            print('validation:',evaluate(model,val_dl,device)); save_checkpoint(ckpt/'latest.pt',model,opt,step,str(Path(a.config)),{'seed':seed,'device':str(device),'tokenizer_path':a.tokenizer,'amp':a.amp})
    save_checkpoint(ckpt/'latest.pt',model,opt,step0+max_steps,str(Path(a.config)),{'seed':seed,'device':str(device),'tokenizer_path':a.tokenizer,'amp':a.amp})
if __name__=='__main__': main()
