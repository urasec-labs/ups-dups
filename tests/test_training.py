import torch

def test_optimizer_step(tiny_model):
 opt=torch.optim.AdamW(tiny_model.parameters(),lr=1e-3); x=torch.randint(0,260,(1,8)); y=torch.randint(0,260,(1,8)); before=tiny_model.token_embedding.weight.detach().clone()
 loss=tiny_model(x,y)['loss']; loss.backward(); torch.nn.utils.clip_grad_norm_(tiny_model.parameters(),1.); opt.step(); assert not torch.equal(before,tiny_model.token_embedding.weight.detach())

def test_deterministic_initialization(tiny_config):
 from model.transformer import UpsDupsLM
 torch.manual_seed(42); a=UpsDupsLM(tiny_config); torch.manual_seed(42); b=UpsDupsLM(tiny_config); assert torch.equal(a.token_embedding.weight,b.token_embedding.weight)
