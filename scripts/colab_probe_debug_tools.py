import shutil
import subprocess
print({name:shutil.which(name) for name in ('gdb','jcmd','jstack','java')})
print(subprocess.run(['py-spy','dump','--help'],capture_output=True,text=True).stdout[:2400])
