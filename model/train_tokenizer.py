"""Train a small byte-level BPE tokenizer from a UTF-8 text corpus."""
import argparse
from pathlib import Path
from model.tokenizer import ByteBPETokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',default='artifacts/tokenizer.json'); p.add_argument('--vocab-size',type=int,default=4096); p.add_argument('--min-frequency',type=int,default=2); a=p.parse_args()
    path=Path(a.input)
    with path.open('r',encoding='utf-8',errors='replace') as f: texts=[line.rstrip('\n') for line in f if line.strip()]
    if not texts: raise SystemExit('input corpus contains no non-empty lines')
    tokenizer=ByteBPETokenizer.train(texts,a.vocab_size,a.min_frequency); tokenizer.save(a.output)
    print(f'trained_vocab_size={tokenizer.vocab_size} documents_or_lines={len(texts)} output={a.output}')
if __name__=='__main__': main()
