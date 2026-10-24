import argparse,torch
from model.config import load_config
from model.transformer import UpsDupsLM
from model.tokenizer import ByteTokenizer,ByteBPETokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--prompt',required=True); p.add_argument('--max-new-tokens',type=int,default=50); p.add_argument('--temperature',type=float,default=.8); p.add_argument('--top-k',type=int,default=40); p.add_argument('--top-p',type=float,default=.95); p.add_argument('--tokenizer',default=None); a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True); path=payload.get('config_path')
    if not path: raise SystemExit('Checkpoint missing config_path metadata.')
    cfg=load_config(path)[1]; model=UpsDupsLM(cfg); model.load_state_dict(payload['model']); model.eval(); tok=ByteBPETokenizer.load(a.tokenizer) if a.tokenizer else ByteTokenizer();
    if tok.vocab_size != cfg.vocab_size: raise SystemExit(f'tokenizer vocab size {tok.vocab_size} does not match model config {cfg.vocab_size}')
    ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long)
    print(tok.decode(model.generate(ids,a.max_new_tokens,a.temperature,a.top_k,a.top_p)[0].tolist()))
if __name__=='__main__': main()
