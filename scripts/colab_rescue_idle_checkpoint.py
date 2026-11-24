import json, subprocess, sys
from pathlib import Path
w=Path('/content/crrl_cache_upgrade')
print('READY', (w/'debugger_ready.json').read_text() if (w/'debugger_ready.json').exists() else (w/'debugger_setup.log').read_text()[-1200:])
rescue='''import sys,json,traceback
from pathlib import Path
p=Path('/content/crrl_cache_upgrade/rescued.json')
try:
    found=[]
    for frame in sys._current_frames().values():
        while frame:
            obj=frame.f_locals.get('self')
            if type(obj).__name__=='PersistentStateSelfplay' and frame.f_code.co_name=='collect': found.append(obj)
            frame=frame.f_back
    assert found, 'No idle collector found'
    trainer=found[0]
    assert trainer.update==949, trainer.update
    p.write_text(json.dumps({'stage':'saving','update':trainer.update}))
    checkpoint=trainer.checkpoint()
    p.write_text(json.dumps({'stage':'saved','update':trainer.update,'checkpoint':str(checkpoint)}))
except BaseException:
    p.write_text(json.dumps({'stage':'error','traceback':traceback.format_exc()}))
'''
(w/'rescue_payload.py').write_text(rescue)
if (w/'debugger_ready.json').exists() and not (w/'gdb.pid').exists():
    assert b'/train_selfplay.py\0' in Path('/proc/26990/cmdline').read_bytes()
    code="exec(compile(open('/content/crrl_cache_upgrade/rescue_payload.py').read(), 'rescue', 'exec'))"
    cmds=['set pagination off','set confirm off','set scheduler-locking off','thread 1','set $rescue_gil = (int)PyGILState_Ensure()', 'call (int)PyRun_SimpleString('+json.dumps(code)+')','call (void)PyGILState_Release($rescue_gil)','detach','quit']
    args=['gdb','-q','-nx','-batch','-p','26990']
    for c in cmds: args+=['-ex',c]
    with (w/'gdb.log').open('w') as log:
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (w/'gdb.pid').write_text(str(p.pid))
    print('RESCUE_PID',p.pid)
for n in ['rescued.json','gdb.log']:
    p=w/n
    if p.exists():print(n,p.read_text()[-2000:])
