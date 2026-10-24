from fastapi.testclient import TestClient
import api.app as api

def test_health_metadata_and_no_fake_response(monkeypatch):
 monkeypatch.setattr(api,'MODEL',None); monkeypatch.setattr(api,'MODEL_STATUS',{'loaded':False,'reason':'test'})
 with TestClient(api.app) as client:
  assert client.get('/health').status_code==200; assert client.get('/v1/tokenizer').json()['vocab_size']==260; assert client.get('/').status_code==200
  r=client.post('/v1/chat',json={'message':'Merhaba'}); assert r.status_code==503 and 'No trained model' in r.json()['detail']

def test_input_validation():
 with TestClient(api.app) as client:
  assert client.post('/v1/chat',json={'message':''}).status_code==422
  assert client.post('/v1/chat',json={'message':'x','max_new_tokens':900}).status_code==422
