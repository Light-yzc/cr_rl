"""CPU-stream an owned raw cache into one binary plus an integrity index.

Never loads a full cache/list of tensors into RAM. Original block files are
removed only with --delete-originals, after every tensor and the complete binary
pass SHA256 readback verification and the source files are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import torch


FORMAT = 'crrl-raw-cache-pack-v1'
BLOCK_PATTERN = re.compile(r'block_[0-9]{6}\.pt\Z')
WORKSPACE = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(data)
    return digest.hexdigest()


def checked_cache(path):
    raw = Path(path)
    cache = raw.resolve(strict=True)
    artifacts = (WORKSPACE/'artifacts').resolve(strict=True)
    if (cache.drive.upper() != 'G:' or cache.name != 'feature_cache' or
            not cache.is_relative_to(artifacts) or raw.is_symlink()):
        raise ValueError('Only a resolved G: workspace artifacts/.../feature_cache directory is accepted')
    return cache


def verify_binary(path,index,*,progress=False):
    if path.stat().st_size != index['bin']['bytes']:
        raise ValueError('Packed binary byte count differs')
    cursor = 0
    with path.open('rb') as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        for number,block in enumerate(index['blocks'],1):
            block_digest = hashlib.sha256()
            for field,record in block['tensors'].items():
                dtype = np.dtype(record['dtype'])
                if (record['offset'] != cursor or record['nbytes'] !=
                        int(np.prod(record['shape'],dtype=np.int64))*dtype.itemsize):
                    raise ValueError('Non-contiguous tensor offsets or invalid shape/byte count')
                array = np.ndarray(tuple(record['shape']),dtype=dtype,buffer=mapping,offset=record['offset'])
                view = memoryview(array).cast('B')
                digest = hashlib.sha256(view).hexdigest()
                block_digest.update(view)
                view.release()
                del array
                if digest != record['sha256']:
                    raise ValueError(f'Packed tensor readback differs: {block["name"]}/{field}')
                cursor += record['nbytes']
            if block_digest.hexdigest() != block['tensor_bytes_sha256']:
                raise ValueError('Concatenated block readback differs')
            if progress and number%1024==0:
                print(json.dumps({'phase':'verify_tensors','blocks':number,'total':len(index['blocks'])}),flush=True)
    if cursor != index['bin']['bytes'] or file_sha256(path) != index['bin']['sha256']:
        raise ValueError('Complete packed binary SHA256 readback differs')


def pack(cache,*,delete_originals=False,progress=True):
    cache = checked_cache(cache)
    protocol_path = cache.parent/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = file_sha256(protocol_path)
    if not protocol.get('completed'):
        raise ValueError('Source raw-cache extraction must be completed before packing')
    files = sorted(cache.glob('block_*.pt'))
    if not files or len(files)!=protocol['extraction']['blocks']:
        raise ValueError('Source block count differs from completed extraction protocol')
    if any(not BLOCK_PATTERN.fullmatch(path.name) or path.is_symlink() or
            path.resolve().parent != cache for path in files):
        raise ValueError('Unexpected original block path; refusing packing/deletion')
    binary,index_path = cache/'raw_cache.bin',cache/'index.json'
    pending = cache/'raw_cache.bin.tmp'
    index_pending = cache/'index.json.tmp'
    if any(path.exists() for path in (binary,index_path,pending,index_pending)):
        raise ValueError('Packed output or pending file exists; inspect it before another pack')
    started = time.perf_counter()
    index = {'format':FORMAT,'source_protocol_sha256':protocol_sha,
        'checkpoint_sha256':protocol['checkpoint_sha256'],
        'parent_manifest_sha256':protocol.get('parent_manifest_sha256'),
        'source_sha256':protocol.get('source_sha256'),
        'bin':{'path':'raw_cache.bin','bytes':0,'sha256':None},'blocks':[],
        'storage':'C-contiguous raw tensor bytes, original numpy dtype.str, no padding',
        'integrity':'Per-tensor and concatenated-block SHA256, plus complete binary SHA256; readback verified',
        'original_files_removed':False}
    whole_digest = hashlib.sha256()
    source_stats = []
    source_bytes = 0
    with pending.open('xb') as stream:
        for number,path in enumerate(files,1):
            before = path.stat()
            payload = torch.load(path,map_location='cpu',weights_only=False)
            block = {'name':path.stem,'metadata':{},'tensors':{}}
            block_digest = hashlib.sha256()
            for key,value in payload.items():
                if not isinstance(value,torch.Tensor):
                    json.dumps(value,allow_nan=False)
                    block['metadata'][key] = value
                    continue
                if value.device.type!='cpu' or value.layout!=torch.strided:
                    raise ValueError('Only dense CPU tensor cache fields are supported')
                array = value.detach().contiguous().numpy()
                if array.dtype.hasobject:
                    raise ValueError('Object arrays cannot be packed')
                view = memoryview(array).cast('B')
                offset = stream.tell()
                stream.write(view)
                whole_digest.update(view)
                block_digest.update(view)
                block['tensors'][key] = {'offset':offset,'nbytes':view.nbytes,
                    'shape':list(array.shape),'dtype':array.dtype.str,'torch_dtype':str(value.dtype),
                    'sha256':hashlib.sha256(view).hexdigest()}
                view.release()
                del array
            after = path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('An original block changed while packing')
            source_stats.append((path,before.st_size,before.st_mtime_ns))
            source_bytes += before.st_size
            block['tensor_bytes_sha256'] = block_digest.hexdigest()
            index['blocks'].append(block)
            del payload,value
            if progress and (number%512==0 or number==len(files)):
                print(json.dumps({'phase':'pack_write','blocks':number,'total':len(files),
                    'raw_mib':stream.tell()/1024**2,'seconds':time.perf_counter()-started}),flush=True)
        stream.flush()
        os.fsync(stream.fileno())
        index['bin'].update(bytes=stream.tell(),sha256=whole_digest.hexdigest())
    verify_binary(pending,index,progress=progress)
    if file_sha256(protocol_path)!=protocol_sha or sorted(cache.glob('block_*.pt'))!=files:
        raise ValueError('Completed source protocol or file list changed')
    # Validate every source before deleting even the first original file.
    for path,size,mtime in source_stats:
        current = path.stat()
        if path.is_symlink() or path.resolve().parent!=cache or (current.st_size,current.st_mtime_ns)!=(size,mtime):
            raise ValueError('Original source changed before committing packed cache')
    index.update(readback_verified=True,source_block_count=len(files),source_file_bytes=source_bytes,
        pack_seconds=time.perf_counter()-started)
    index_pending.write_text(json.dumps(index,separators=(',',':'),allow_nan=False),encoding='utf-8')
    pending.replace(binary)
    index_pending.replace(index_path)
    if delete_originals:
        for path,_,_ in source_stats:
            # Native Python unlink, exact validated files, no recursive deletion.
            path.unlink()
        index['original_files_removed'] = True
        index['deleted_original_files'] = len(source_stats)
        index_pending.write_text(json.dumps(index,separators=(',',':'),allow_nan=False),encoding='utf-8')
        index_pending.replace(index_path)
    receipt = {'format':FORMAT,'cache':str(cache),'blocks':len(files),'tensor_bytes':index['bin']['bytes'],
        'binary_sha256':index['bin']['sha256'],'index_bytes':index_path.stat().st_size,
        'original_files_removed':delete_originals,'source_file_bytes':source_bytes,
        'seconds':time.perf_counter()-started,'readback_verified':True,
        'index_path':str(index_path),'binary_path':str(binary)}
    (cache.parent/'pack_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def self_test():
    scratch = WORKSPACE/'artifacts/bc_capacity_20261111/tmp'
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pack-cpu-',dir=scratch) as temporary:
        root = Path(temporary)
        cache = root/'feature_cache'
        cache.mkdir()
        for i in (1,2):
            torch.save({'partition':'train','replay_id':i,'seat':0,
                'context':torch.arange(6,dtype=torch.float16).reshape(2,3),
                'entity_mask':torch.tensor([[True,False,True]]),
                'tiles':torch.tensor([5,31],dtype=torch.int64)},cache/f'block_{i:06d}.pt')
        (root/'protocol.json').write_text(json.dumps({'completed':True,'extraction':{'blocks':2},'checkpoint_sha256':'cpu-fixture'}))
        result=pack(cache,delete_originals=True,progress=False)
        index=json.loads((cache/'index.json').read_text())
        assert result['readback_verified'] and not list(cache.glob('block_*.pt'))
        assert index['blocks'][0]['tensors']['context']['dtype']=='<f2'
        assert index['blocks'][0]['tensors']['entity_mask']['dtype']=='|b1'
        verify_binary(cache/'raw_cache.bin',index)
        record=index['blocks'][0]['tensors']['tiles']
        values=np.memmap(cache/'raw_cache.bin',mode='r',dtype=record['dtype'],
            offset=record['offset'],shape=tuple(record['shape']))
        assert values.tolist()==[5,31]
        del values
        return {'passed':True,'device':'cpu','preserves_fp16_bool_int64':True,
            'unaligned_no_padding_offset_supported':True,'all_tensor_and_file_readback_verified':True,
            'originals_deleted_only_after_verification':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--delete-originals',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        if args.cache is None:
            parser.error('--cache is required')
        print(json.dumps(pack(args.cache,delete_originals=args.delete_originals)),flush=True)
