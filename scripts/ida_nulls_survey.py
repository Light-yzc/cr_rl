"""IDA batch survey of the local test client's ELF; no device operations.

Run with -A -a: automatic whole-program analysis is deliberately disabled.
"""
import json
from pathlib import Path
import ida_idaapi
import ida_nalt
import ida_segment
import ida_bytes
import ida_name
import idautils
import ida_pro
import ida_ua
import idc

out = Path(ida_nalt.get_input_file_path()).parent / 'ida_survey.json'
segments=[]
for address in idautils.Segments():
    s=ida_segment.getseg(address)
    segments.append({'name':ida_segment.get_segm_name(s),'start':hex(s.start_ea),
                     'end':hex(s.end_ea),'permissions':s.perm})
imports=[]
for i in range(ida_nalt.get_import_module_qty()):
    module=ida_nalt.get_import_module_name(i)
    def collect(ea,name,ordinal):
        imports.append({'address':hex(ea),'name':name,'ordinal':ordinal,'module':module})
        return True
    ida_nalt.enum_import_names(i,collect)
# Limit output to symbols that may help identify the observation pipeline.
names=[{'address':hex(ea),'name':name} for ea,name in idautils.Names()
       if any(k in name.lower() for k in ('battle','elixir','player','card','jni_onload'))][:500]
out.write_text(json.dumps({'scope':'loader-only; no full auto-analysis',
    'segments':segments,'imports':imports,'selected_names':names},indent=2),encoding='utf-8')
# Decode only the few hundred instructions around independently found candidates.
disassembly=[]
for start,end in [(0xf126d0,0xf129e0),(0xeb9b70,0xeb9c80)]:
    for ea in range(start,end,4):
        ida_ua.create_insn(ea)
        disassembly.append(f'{ea:08x}: {idc.generate_disasm_line(ea,0)}')
(out.parent/'candidate_disassembly.txt').write_text('\n'.join(disassembly),encoding='utf-8')
ida_pro.qexit(0)
