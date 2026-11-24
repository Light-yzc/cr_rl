"""Bound local Colab clients including launcher descendants on Windows."""
import os
import signal
import subprocess


def terminate_tree(process):
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
    else:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait()


def run_bounded(command, *, timeout, env=None, cwd=None):
    flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=env,cwd=cwd,creationflags=flags,start_new_session=os.name!='nt')
    try:
        out,err=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_tree(process)
        out,err=process.communicate()
        raise subprocess.TimeoutExpired(command,timeout,output=out,stderr=err) from None
    return subprocess.CompletedProcess(command,process.returncode,out,err)
