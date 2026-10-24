from model.tokenizer import ByteBPETokenizer

def test_bpe_train_roundtrip(tmp_path):
    texts=['Türkçe dil modeli merhaba','Türkçe dil modeli merhaba','Türkçe dil modeli öğreniyor']
    tok=ByteBPETokenizer.train(texts,vocab_size=300,min_frequency=2)
    sample='Türkçe dil modeli merhaba 🌍'
    assert tok.decode(tok.encode(sample))==sample
    path=tmp_path/'tokenizer.json'; tok.save(path); loaded=ByteBPETokenizer.load(path)
    assert loaded.vocab_size==tok.vocab_size
    assert loaded.encode(sample)==tok.encode(sample)

def test_bpe_special_tokens():
    tok=ByteBPETokenizer.train(['aaaa aaaa','aaaa aaaa'],vocab_size=270,min_frequency=2)
    ids=tok.encode('abc',add_bos=True,add_eos=True)
    assert ids[0]==tok.BOS and ids[-1]==tok.EOS
    assert tok.decode(ids)=='abc'
