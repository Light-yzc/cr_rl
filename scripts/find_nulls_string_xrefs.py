"""Bounded ELF AArch64 ADRP/ADD candidate references for manual review.

Candidates are not validated function boundaries or live game field offsets.
"""
import argparse
import json
from pathlib import Path
import struct


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--strings',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    b=a.library.read_bytes()
    if b[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',b,18)[0]!=183:
        raise ValueError('Expected little-endian ELF64 AArch64')
    phoff=struct.unpack_from('<Q',b,32)[0]
    phsize,phnum=struct.unpack_from('<HH',b,54)
    loads=[]
    for i in range(phnum):
        typ,flags,off,va,_,size,memsz,_=struct.unpack_from('<IIQQQQQQ',b,phoff+i*phsize)
        if typ==1:loads.append((off,va,size,flags))
    targets={}
    for row in json.loads(a.strings.read_text()):
        if not any(k in row['text'].lower() for k in ('current_hitpoints','currentelixir','current_elixir','battleclock','spendelixir')):
            continue
        off=int(row['file_offset'],16)
        for start,va,size,_ in loads:
            if start<=off<start+size:targets[va+off-start]=row['text']
    pages={v&~4095 for v in targets}
    refs=[]
    for start,va,size,flags in loads:
        if not flags&1:continue
        for off in range(start,start+size-4,4):
            word=struct.unpack_from('<I',b,off)[0]
            if word&0x9f000000!=0x90000000:continue
            imm=((word>>5)&0x7ffff)<<2 | ((word>>29)&3)
            if imm&(1<<20):imm-=1<<21
            page=((va+off-start)&~4095)+(imm<<12)
            if page not in pages:continue
            reg=word&31
            for later in range(off+4,min(off+36,start+size-3),4):
                ins=struct.unpack_from('<I',b,later)[0]
                if ins&0x7f000000!=0x11000000 or (ins>>5)&31!=reg:continue
                delta=(ins>>10)&0xfff
                if ins&(1<<22):delta<<=12
                target=page+delta
                if target in targets:
                    refs.append({'adrp':hex(va+off-start),'add':hex(va+later-start),
                                 'target':hex(target),'text':targets[target]})
    result={'scope':'static heuristic candidates only','targets':{hex(k):v for k,v in targets.items()},'references':refs}
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
