"""Run a local model on a tiny Turkish smoke suite. Metrics are exact/normalized match only."""
import argparse,json,re,torch
from pathlib import Path
from model.config import load_config
from model.transformer import UpsDupsLM
from model.tokenizer import ByteTokenizer,ByteBPETokenizer

def norm(s): return re.sub(r'\s+',' ',s.casefold()).strip().strip('.,!?;:')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--suite',default='evaluation/turkish_suite.jsonl'); p.add_argument('--tokenizer',default=None); p.add_argument('--max-new-tokens',type=int,default=32); p.add_argument('--output',default='reports/turkish_eval.json'); a=p.parse_args()
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True); cfg_path=payload.get('config_path')
 if not cfg_path: raise SystemExit('checkpoint lacks config_path metadata')
 _,cfg=load_config(cfg_path); model=UpsDupsLM(cfg); model.load_state_dict(payload['model']); model.eval()
 tok=ByteBPETokenizer.load(a.tokenizer) if a.tokenizer else ByteTokenizer()
 if tok.vocab_size != cfg.vocab_size: raise SystemExit(f'tokenizer vocab size {tok.vocab_size} does not match model config {cfg.vocab_size}')
 rows=[]
 with open(a.suite,encoding='utf-8') as f: cases=[json.loads(line) for line in f if line.strip()]
 for case in cases:
  ids=torch.tensor([tok.encode(case['prompt'])],dtype=torch.long)
  max_new=min(a.max_new_tokens,max(1,cfg.max_seq_len-1))
  truncated=ids.shape[1] > cfg.max_seq_len-max_new
  if truncated: ids=ids[:,-(cfg.max_seq_len-max_new):]
  with torch.no_grad(): out=model.generate(ids,max_new_tokens=max_new,temperature=1.0,top_k=1,top_p=1.0)
  prediction=tok.decode(out[0].tolist()[ids.shape[1]:])
  rows.append({**case,'prompt_truncated':truncated,'prediction':prediction,'normalized_match':norm(prediction)==norm(case['reference'])})
 result={'warning':'Tiny smoke suite; not a validated benchmark and not evidence of general Turkish capability.','cases':len(rows),'normalized_match_count':sum(r['normalized_match'] for r in rows),'normalized_match_rate':sum(r['normalized_match'] for r in rows)/max(1,len(rows)),'results':rows}
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
