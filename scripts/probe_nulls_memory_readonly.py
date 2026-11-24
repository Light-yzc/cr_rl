"""Bounded read-only permission/profile probe; never injects or writes memory."""
import argparse
import json
import re
import struct
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adb', default='G:/sdk/platform-tools/adb.exe')
    p.add_argument('--serial', required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    prefix = [args.adb, '-s', args.serial]
    def command(*argv):
        return subprocess.check_output(prefix + list(argv), timeout=15)
    pid = int(command('shell', 'pidof', 'nullsroyale.rel.free').strip())
    maps = command('shell', 'cat', f'/proc/{pid}/maps').decode()
    regions = []
    for line in maps.splitlines():
        match = re.match(r'([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+([0-9a-f]+)', line)
        if match:
            start, end, perms, offset = match.groups()
            regions.append((int(start,16),int(end,16),perms,int(offset,16),line))
    libs = [r for r in regions if r[4].endswith('/libg.so')]
    base = min(r[0]-r[3] for r in libs)
    def readable(address, size):
        return any(a <= address and address+size <= b and 'r' in perm for a,b,perm,_,_ in regions)
    def read(address,size):
        if not readable(address,size):
            raise ValueError('Profile address is not in a readable mapping')
        data = command('exec-out',f'dd if=/proc/{pid}/mem bs=1 skip={address} count={size} 2>/dev/null')
        if len(data) != size:
            raise ValueError(f'Short memory read: {len(data)}/{size}')
        return data
    result = {'pid':pid,'libg_base':hex(base),'root_shell':command('shell','id').decode().strip()}
    try:
        result['elf_magic'] = read(base,4).hex()
        # Existing reader's version-specific ARM_MANAGER_GLOBAL; test only.
        address = base + 0x19BBDD8
        manager = struct.unpack('<Q',read(address,8))[0]
        result.update(old_profile_manager=hex(manager),manager_readable=readable(manager,8))
        chain=[]
        result['old_profile_chain']=chain
        for name, offset in [('context',0x248),('battle',0x90),('hp_state',0xA8),('registry',0x08),('collection',0x40)]:
            address=manager+offset
            manager=struct.unpack('<Q',read(address,8))[0]
            chain.append({'name':name,'value':hex(manager),'readable':readable(manager,8)})
    except Exception as error:
        result['error']=str(error)
    result['captured_unix']=time.time()
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        stamp=str(time.time_ns())
        (args.out/(stamp+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        (args.out/(stamp+'.png')).write_bytes(command('exec-out','screencap','-p'))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
