import torch,pytest
from model.tokenizer import ByteTokenizer
from model.attention import apply_rope,RotaryEmbedding

def test_tokenizer_roundtrip():
 t=ByteTokenizer(); s='Türkçe, merhaba 🌍'; assert t.decode(t.encode(s))==s

def test_forward_loss_backward(tiny_model):
 x=torch.randint(0,260,(2,12)); y=torch.randint(0,260,(2,12)); out=tiny_model(x,y)
 assert out['logits'].shape==(2,12,260); assert out['loss'].ndim==0 and torch.isfinite(out['loss']); out['loss'].backward(); assert tiny_model.token_embedding.weight.grad is not None

def test_sequence_limit(tiny_model):
 with pytest.raises(ValueError): tiny_model(torch.zeros((1,33),dtype=torch.long))
def test_bad_input_shape(tiny_model):
 with pytest.raises(ValueError): tiny_model(torch.zeros((3,),dtype=torch.long))
def test_rope_shapes():
 x=torch.randn(2,4,8,16); cos,sin=RotaryEmbedding(16)(x); q,k=apply_rope(x,x,cos,sin); assert q.shape==x.shape and k.shape==x.shape
