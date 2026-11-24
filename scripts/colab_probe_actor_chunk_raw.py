import sys,time
sys.path.insert(0,'scripts')
from download_latest_actor import RemoteFiles
r=RemoteFiles('cr-train-l4-1117');part=r.receipt()['parts'][0];print('PART_SIZE',part['bytes'],flush=True)
t=time.time()
try:
 with r.get(part['remote_path'],stream=True) as response:
  print('RAW_PART',response.status_code,response.headers.get('Content-Length'),response.headers.get('Content-Type'),flush=True)
  response.raise_for_status();block=next(response.iter_content(1024));print('FIRST_KIB',len(block),'SECONDS',time.time()-t,flush=True)
except Exception as e:print('RAW_PART_FAILED',type(e).__name__,time.time()-t,flush=True)
