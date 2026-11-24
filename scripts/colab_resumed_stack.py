import subprocess
r=subprocess.run(['py-spy','dump','--pid','49101'],capture_output=True,text=True,timeout=15)
print(r.stdout[:3600])
