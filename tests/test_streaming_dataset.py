from training.dataset import StreamingTokenBlockDataset

def test_streaming_dataset_blocks(tmp_path):
    p=tmp_path/'shard-00001.txt'; p.write_text('Hello world\nTürkçe metin\n',encoding='utf-8')
    ds=StreamingTokenBlockDataset([p],seq_len=4)
    rows=list(ds)
    assert rows
    x,y=rows[0]
    assert x.shape==(4,) and y.shape==(4,)
    assert (x[1:]==y[:-1]).all()
