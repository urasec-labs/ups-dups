import torch

def sample_next_token(logits,history,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.):
    if temperature<=0: raise ValueError('temperature must be > 0')
    if top_k<0: raise ValueError('top_k must be >= 0')
    if not 0<top_p<=1: raise ValueError('top_p must be in (0, 1]')
    if repetition_penalty<=0: raise ValueError('repetition_penalty must be > 0')
    scores=logits.clone()/temperature
    if repetition_penalty!=1.:
        for i in range(scores.shape[0]):
            ids=history[i].unique(); vals=scores[i,ids]
            scores[i,ids]=torch.where(vals<0,vals*repetition_penalty,vals/repetition_penalty)
    if top_k:
        k=min(top_k,scores.shape[-1]); threshold=torch.topk(scores,k,dim=-1).values[:,-1,None]; scores=scores.masked_fill(scores<threshold,float('-inf'))
    if top_p<1.:
        sorted_logits,idx=torch.sort(scores,descending=True,dim=-1); probs=torch.softmax(sorted_logits,dim=-1); cumulative=probs.cumsum(-1)
        remove=cumulative-probs>=top_p; sorted_logits=sorted_logits.masked_fill(remove,float('-inf'))
        filtered=torch.full_like(scores,float('-inf')); filtered.scatter_(1,idx,sorted_logits); scores=filtered
    probs=torch.softmax(scores,dim=-1)
    if not torch.isfinite(probs).all(): raise RuntimeError('sampling distribution is not finite')
    return torch.multinomial(probs,1)
