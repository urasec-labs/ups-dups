import os,logging,time,codecs
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse,StreamingResponse
from pydantic import BaseModel,Field
from model.config import load_config
from model.transformer import UpsDupsLM
from model.tokenizer import ByteTokenizer,ByteBPETokenizer

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s'); log=logging.getLogger('upsdups')
@asynccontextmanager
async def lifespan(app):
    load_model_from_env()
    yield

app=FastAPI(title='Ups-dups Enterprise local API',version='0.1.0',lifespan=lifespan); MODEL=None; TOKENIZER=ByteTokenizer(); MODEL_STATUS={'loaded':False,'reason':'No checkpoint configured'}
class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=12000)
    max_new_tokens:int=Field(default=128,ge=1,le=512)
    temperature:float=Field(default=.8,gt=0.,le=2.)
    top_k:int=Field(default=40,ge=0,le=8192)
    top_p:float=Field(default=.95,gt=0.,le=1.)
    repetition_penalty:float=Field(default=1.,gt=0.,le=3.)

def load_model_from_env():
    global MODEL,MODEL_STATUS,TOKENIZER
    ckpt=os.environ.get('UPS_DUPS_CHECKPOINT')
    if not ckpt: MODEL_STATUS={'loaded':False,'reason':'Set UPS_DUPS_CHECKPOINT to a trained checkpoint to enable generation.'}; return
    try:
        payload=torch.load(ckpt,map_location='cpu',weights_only=True); cfg_path=payload.get('config_path')
        if not cfg_path or not Path(cfg_path).exists(): raise ValueError('checkpoint config_path missing or unavailable')
        cfg=load_config(cfg_path)[1]; tokenizer_path=os.environ.get('UPS_DUPS_TOKENIZER'); tokenizer=ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else ByteTokenizer()
        if tokenizer.vocab_size != cfg.vocab_size: raise ValueError(f'tokenizer vocab size {tokenizer.vocab_size} does not match model config {cfg.vocab_size}')
        model=UpsDupsLM(cfg); model.load_state_dict(payload['model'],strict=True); model.eval(); MODEL=model; TOKENIZER=tokenizer
        MODEL_STATUS={'loaded':True,'reason':None,'checkpoint':str(Path(ckpt).resolve()),'step':payload.get('step')}
    except Exception as e:
        MODEL=None; MODEL_STATUS={'loaded':False,'reason':f'Checkpoint load failed: {type(e).__name__}: {e}'}; log.exception('checkpoint load failed')
@app.get('/health')
def health(): return {'status':'ok','model_loaded':MODEL is not None,'model_status':MODEL_STATUS}
@app.get('/v1/model')
def model_info(): return {'name':'Ups-dups Enterprise','model_loaded':MODEL is not None,'status':MODEL_STATUS,'tokenizer':type(TOKENIZER).__name__,'tokenizer_vocab_size':TOKENIZER.vocab_size,'external_llm_api':False}
@app.get('/v1/tokenizer')
def tokenizer_info(): return {'type':type(TOKENIZER).__name__,'vocab_size':TOKENIZER.vocab_size,'special_tokens':{'pad':TOKENIZER.PAD,'bos':TOKENIZER.BOS,'eos':TOKENIZER.EOS,'unk':TOKENIZER.UNK}}
@app.get('/',response_class=HTMLResponse)
def index():
    return HTMLResponse('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ups-dups Enterprise</title><style>body{font:16px system-ui;background:#0b1020;color:#eaf0ff;max-width:800px;margin:40px auto;padding:0 18px}h1{letter-spacing:-.04em}textarea{width:100%;box-sizing:border-box;background:#151d33;color:#fff;border:1px solid #34415e;border-radius:12px;padding:14px;min-height:110px}button{background:#b8e986;color:#14200c;border:0;border-radius:10px;padding:12px 18px;font-weight:700;cursor:pointer;margin-top:10px}pre{white-space:pre-wrap;background:#121a2c;padding:16px;border-radius:12px;min-height:100px}.muted{color:#9ba9c7}</style><h1>Ups-dups Enterprise</h1><p class="muted">Local model interface. Requires a trained checkpoint; no external LLM API is used.</p><textarea id="msg" maxlength="12000" placeholder="Write a prompt…"></textarea><br><button onclick="send()">Generate</button><pre id="out">Checking model status…</pre><script>async function send(){let o=document.getElementById('out');o.textContent='Generating…';try{let r=await fetch('/v1/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:document.getElementById('msg').value})});let d=await r.json();if(!r.ok)throw Error(d.detail||r.statusText);o.textContent=d.text}catch(e){o.textContent='Error: '+e.message}}fetch('/health').then(r=>r.json()).then(d=>{document.getElementById('out').textContent=d.model_loaded?'Model loaded. Ready.':'Model unavailable: '+d.model_status.reason})</script></html>''')
def _generate(req):
    ids=TOKENIZER.encode(req.message)
    if len(ids)>=MODEL.config.max_seq_len: ids=ids[-(MODEL.config.max_seq_len-1):]
    if req.max_new_tokens > MODEL.config.max_seq_len-len(ids): raise HTTPException(status_code=422,detail=f'max_new_tokens exceeds remaining context budget ({MODEL.config.max_seq_len-len(ids)} tokens)')
    x=torch.tensor([ids],dtype=torch.long)
    with torch.inference_mode(): out=MODEL.generate(x,req.max_new_tokens,req.temperature,req.top_k,req.top_p,req.repetition_penalty)
    return TOKENIZER.decode(out[0,len(ids):].tolist())
@app.post('/v1/chat')
def chat(req:ChatRequest):
    if MODEL is None: raise HTTPException(status_code=503,detail='No trained model is loaded. Set UPS_DUPS_CHECKPOINT to a valid checkpoint. No fake response will be returned.')
    start=time.perf_counter()
    try: result=_generate(req); return {'text':result,'generated_tokens':len(TOKENIZER.encode(result)),'elapsed_seconds':round(time.perf_counter()-start,4),'model':'Ups-dups Enterprise'}
    except Exception as e: log.exception('generation failed'); raise HTTPException(status_code=500,detail=f'Generation failed: {type(e).__name__}')
@app.post('/v1/chat/stream')
def chat_stream(req:ChatRequest):
    if MODEL is None: raise HTTPException(status_code=503,detail='No trained model is loaded. Set UPS_DUPS_CHECKPOINT to a valid checkpoint.')
    def events():
        import json
        try:
            ids=TOKENIZER.encode(req.message)
            ids=ids[-(MODEL.config.max_seq_len-1):]
            if req.max_new_tokens > MODEL.config.max_seq_len-len(ids):
                yield 'data: '+json.dumps({'error':'max_new_tokens_exceeds_context'})+'\n\n'; return
            context=torch.tensor([ids],dtype=torch.long)
            decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
            with torch.inference_mode():
                result=MODEL(context,use_cache=True); logits=result['logits'][:,-1,:]; cache=result['past_key_values']
                for index in range(req.max_new_tokens):
                    from inference.sampling import sample_next_token
                    nxt=sample_next_token(logits,context,req.temperature,req.top_k,req.top_p,req.repetition_penalty)
                    token=int(nxt.item()); context=torch.cat([context,nxt],dim=1)
                    if index+1 < req.max_new_tokens:
                        result=MODEL(nxt,past_key_values=cache,use_cache=True); logits=result['logits'][:,-1,:]; cache=result['past_key_values']
                    if token <= 255:
                        chunk=decoder.decode(bytes([token]),final=False)
                        if chunk: yield 'data: '+json.dumps({'text':chunk},ensure_ascii=False)+'\n\n'
                    elif token==TOKENIZER.EOS:
                        break
                tail=decoder.decode(b'',final=True)
                if tail: yield 'data: '+json.dumps({'text':tail},ensure_ascii=False)+'\n\n'
            yield 'data: [DONE]\n\n'
        except Exception:
            log.exception('streaming generation failed')
            yield 'data: '+json.dumps({'error':'generation_failed'})+'\n\n'
    return StreamingResponse(events(),media_type='text/event-stream')
