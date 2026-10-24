"""Tokenizers for Ups-dups. ByteTokenizer is dependency-free; ByteBPETokenizer learns byte-pair merges."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

class ByteTokenizer:
    PAD, BOS, EOS, UNK = 256, 257, 258, 259
    vocab_size = 260
    def encode(self, text: str, add_bos=False, add_eos=False):
        if not isinstance(text, str): raise TypeError('text must be str')
        ids = list(text.encode('utf-8', errors='replace'))
        if add_bos: ids.insert(0, self.BOS)
        if add_eos: ids.append(self.EOS)
        return ids
    def decode(self, ids, skip_special_tokens=True):
        raw = bytearray(); specials = {self.PAD, self.BOS, self.EOS, self.UNK}
        for token in ids:
            token = int(token)
            if 0 <= token <= 255: raw.append(token)
            elif token in specials and not skip_special_tokens: raw.extend(f'<{token}>'.encode())
        return raw.decode('utf-8', errors='replace')

class ByteBPETokenizer:
    """Small trainable byte-level BPE. It is a research implementation, not a fast production tokenizer."""
    PAD, BOS, EOS, UNK = 256, 257, 258, 259
    def __init__(self, merges=None):
        self.merges = [tuple(map(int, pair)) for pair in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = 260 + len(self.merges)
    @classmethod
    def train(cls, texts, vocab_size=512, min_frequency=2):
        if vocab_size < 260: raise ValueError('vocab_size must be >= 260')
        sequences = [list(str(t).encode('utf-8')) for t in texts if str(t)]
        merges = []
        while 260 + len(merges) < vocab_size:
            counts = Counter()
            for seq in sequences: counts.update(zip(seq, seq[1:]))
            if not counts: break
            pair, freq = counts.most_common(1)[0]
            if freq < min_frequency: break
            merges.append(pair)
            new_id = 260 + len(merges) - 1
            replaced = []
            for seq in sequences:
                out=[]; i=0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair: out.append(new_id); i += 2
                    else: out.append(seq[i]); i += 1
                replaced.append(out)
            sequences = replaced
        return cls(merges)
    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str): raise TypeError('text must be str')
        seq = list(text.encode('utf-8', errors='replace'))
        while len(seq) > 1:
            candidates = [(self.ranks[p], i, p) for i, p in enumerate(zip(seq, seq[1:])) if p in self.ranks]
            if not candidates: break
            _, _, pair = min(candidates)
            token = 260 + self.ranks[pair]; out=[]; i=0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair: out.append(token); i += 2
                else: out.append(seq[i]); i += 1
            seq = out
        if add_bos: seq.insert(0, self.BOS)
        if add_eos: seq.append(self.EOS)
        return seq
    def decode(self, ids, skip_special_tokens=True):
        expansions = {260+i: pair for i, pair in enumerate(self.merges)}
        raw=[]
        def expand(token):
            if token in expansions:
                a,b=expansions[token]; expand(a); expand(b)
            elif 0 <= token <= 255: raw.append(token)
        for item in ids:
            token=int(item)
            if token in (self.PAD,self.BOS,self.EOS,self.UNK):
                if not skip_special_tokens: raw.extend(f'<{token}>'.encode())
            else: expand(token)
        return bytes(raw).decode('utf-8', errors='replace')
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'format':'ups-dups-byte-bpe-v1','merges':self.merges}, indent=2), encoding='utf-8')
    @classmethod
    def load(cls, path):
        obj=json.loads(Path(path).read_text(encoding='utf-8'))
        if obj.get('format') != 'ups-dups-byte-bpe-v1' or not isinstance(obj.get('merges'), list): raise ValueError('invalid BPE tokenizer file')
        return cls(obj['merges'])
