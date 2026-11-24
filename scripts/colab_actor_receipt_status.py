from colab_cli.common import state
from urllib.parse import quote
import requests,json
s=state.get_session(state.resolve_session('cr-train-l4-1117'))
p='/content/drive/MyDrive/cr_rl_ppo/training/longrun/latest.json'
try:
 r=requests.get(s.url.rstrip('/')+'/files/'+quote(p.strip('/'),safe='/'),params={'authuser':'0','colab-runtime-proxy-token':s.token},timeout=(10,20));r.raise_for_status();print(r.text)
except Exception as e:print('ERROR_TYPE',type(e).__name__)
