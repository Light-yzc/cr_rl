from colab_cli.common import state
from urllib.parse import quote
import requests,json
try:
 s=state.get_session(state.resolve_session('cr-train-l4-1117'))
 p='/content/drive/MyDrive/cr_rl_ppo/training/longrun/latest.json'
 with requests.get(s.url.rstrip('/')+'/files/'+quote(p.strip('/'),safe='/'),params={'authuser':'0','colab-runtime-proxy-token':s.token},headers={'Range':'bytes=0-0'},stream=True,timeout=(10,20)) as r:
  print('FILE_ROUTE',r.status_code,'RANGE',r.headers.get('Content-Range'),'LENGTH',r.headers.get('Content-Length'))
except Exception as e:print('TRANSPORT_ERROR',type(e).__name__)
