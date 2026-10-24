"""Split a prepared UTF-8 corpus into bounded-size shards without loading it all into RAM."""
import argparse, hashlib, json
from pathlib import Path

def shard(input_path, output_dir, max_bytes=64*1024*1024):
    if max_bytes < 1024: raise ValueError('max_bytes must be at least 1024')
    input_path=Path(input_path); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    files=[]; idx=0; size=0; digest=hashlib.sha256(); out=None
    def open_next(i): return (output_dir/f'shard-{i:05d}.txt').open('wb')
    try:
        with input_path.open('rb') as source:
            for line in source:
                if out is None or (size and size+len(line)>max_bytes):
                    if out: out.close()
                    idx+=1; out=open_next(idx); files.append({'path':f'shard-{idx:05d}.txt','bytes':0,'sha256':None}); size=0; digest=hashlib.sha256()
                out.write(line); size+=len(line); files[-1]['bytes']+=len(line); digest.update(line); files[-1]['sha256']=digest.hexdigest()
    finally:
        if out: out.close()
    manifest={'input_file':input_path.name,'max_bytes_per_shard':max_bytes,'shards':files,'warning':'Preserve the original source/license manifest; sharding does not grant data rights.'}
    (output_dir/'shard_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--max-bytes',type=int,default=64*1024*1024); a=p.parse_args(); print(json.dumps(shard(a.input,a.output_dir,a.max_bytes),indent=2))
if __name__=='__main__': main()
