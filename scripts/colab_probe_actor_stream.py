import sys,json
sys.path.insert(0,'scripts')
from download_latest_actor import RemoteFiles
r=RemoteFiles('cr-train-l4-1117');receipt=r.receipt();print('RECEIPT',receipt)
for size in ('bytes=0-1023','bytes=0-8388607','bytes=0-'):
 try:
  with r.get(receipt['remote_path'],stream=True,headers={'Range':size}) as response:
   print('RANGE_PROBE',size,response.status_code,{k:response.headers.get(k) for k in ('Content-Length','Content-Range','Content-Type')})
   if response.status_code==206:print('FIRST_BYTES',len(next(response.iter_content(64))))
 except Exception as e:print('ERROR',type(e).__name__)
