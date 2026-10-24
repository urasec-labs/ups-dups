"""Small-corpus map dataset and bounded-memory streaming text-shard dataset."""
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, IterableDataset
from model.tokenizer import ByteTokenizer, ByteBPETokenizer

class TokenBlockDataset(Dataset):
    def __init__(self,path,seq_len,split='train',val_fraction=.05,seed=1337,tokenizer_path=None):
        if seq_len < 1: raise ValueError('seq_len must be positive')
        if split not in {'train','val','all'}: raise ValueError("split must be 'train', 'val', or 'all'")
        if split != 'all' and not 0 < val_fraction < .5: raise ValueError('val_fraction must be in (0, .5)')
        text=Path(path).read_text(encoding='utf-8',errors='replace')
        docs=[d.strip() for d in text.split('\n\n') if d.strip()]
        if len(docs)<2: docs=[d.strip() for d in text.splitlines() if d.strip()]
        if not docs: raise ValueError('dataset must contain at least one non-empty document')
        if split=='all': selected=docs
        else:
            if len(docs)<2: raise ValueError('dataset needs at least two non-empty documents/lines for split')
            rng=random.Random(seed); rng.shuffle(docs); n_val=max(1,int(len(docs)*val_fraction))
            selected=docs[n_val:] if split=='train' else docs[:n_val]
        tok=ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else ByteTokenizer(); ids=[]
        for doc in selected: ids.extend(tok.encode(doc,add_eos=True))
        self.tokens=torch.tensor(ids,dtype=torch.long); self.seq_len=int(seq_len)
        if len(self.tokens)<self.seq_len+1: raise ValueError(f'{split} split has {len(self.tokens)} tokens; needs at least {self.seq_len+1}')
    def __len__(self): return (len(self.tokens)-1)//self.seq_len
    def __getitem__(self,idx):
        if idx<0 or idx>=len(self): raise IndexError(idx)
        start=idx*self.seq_len; chunk=self.tokens[start:start+self.seq_len+1]; return chunk[:-1],chunk[1:]

class StreamingTokenBlockDataset(IterableDataset):
    """Read UTF-8 text shards lazily. Shards should already be split by corpus source."""
    def __init__(self,paths,seq_len,tokenizer_path=None,add_eos=True):
        super().__init__()
        if seq_len<1: raise ValueError('seq_len must be positive')
        if isinstance(paths,(str,Path)): paths=[paths]
        self.paths=[Path(p) for p in paths]
        if not self.paths or any(not p.is_file() for p in self.paths): raise ValueError('all shard paths must exist and be files')
        self.seq_len=int(seq_len); self.add_eos=bool(add_eos)
        self.tokenizer=ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else ByteTokenizer()
    def __iter__(self):
        worker=torch.utils.data.get_worker_info(); paths=self.paths if worker is None else self.paths[worker.id::worker.num_workers]
        buf=[]
        for path in paths:
            with path.open('r',encoding='utf-8',errors='replace') as stream:
                for line in stream:
                    text=line.strip()
                    if not text: continue
                    buf.extend(self.tokenizer.encode(text))
                    if self.add_eos: buf.append(self.tokenizer.EOS)
                    while len(buf)>=self.seq_len+1:
                        chunk=buf[:self.seq_len+1]; del buf[:self.seq_len]
                        yield torch.tensor(chunk[:-1],dtype=torch.long),torch.tensor(chunk[1:],dtype=torch.long)
