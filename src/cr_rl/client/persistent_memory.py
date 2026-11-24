"""Persistent read-only ADB page transport; implements the existing MemoryBackend."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import base64
import errno
import subprocess

from .nulls_reader import BoundedAdbMemReader, _PTR_MASK, is_ptr


class UnmappedGamePageError(OSError):
    """A framed response reports a disappeared/invalid page; snapshot is unusable."""


class PersistentAdbMemReader(BoundedAdbMemReader):
    def begin_poll(self):
        super().begin_poll()
        self.poll_exchanges=0

    def __init__(self, serial, pid, *, adb='adb', helper='/data/local/tmp/crrl-readonly-pages',
                 max_poll_bytes=1024*1024, timeout=3.):
        if not str(pid).isdigit() or not helper.startswith('/data/local/tmp/') or any(c in helper for c in " '\n\r\t;$`"):
            raise ValueError('Invalid helper path or PID')
        super().__init__(serial,pid,adb=adb,max_poll_bytes=max_poll_bytes)
        self.timeout=timeout
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='nulls-readonly')
        self._process=subprocess.Popen([adb,'-s',serial,'shell','-T',f'{helper} {pid}'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            line=self._bounded(self._line)
            if not line.startswith(b'READY '):raise ValueError(f'Invalid transport handshake: {line[:60]!r}')
            status=int(line.split()[1])
            if status: raise OSError(status,'Cannot open game memory read-only')
        except Exception:
            self.close();raise

    def _exact(self,n):
        data=bytearray()
        while len(data)<n:
            part=self._process.stdout.read(n-len(data))
            if not part:raise EOFError('Read-only transport disconnected')
            data.extend(part)
        return bytes(data)

    def _line(self):
        line=self._process.stdout.readline(8192)
        if not line.endswith(b'\n'):raise EOFError('Incomplete read-only transport line')
        return line.rstrip(b'\r\n')

    def _bounded(self,call):
        future=self._executor.submit(call)
        try:return future.result(timeout=self.timeout)
        except TimeoutError:
            self.close()
            raise TimeoutError('Read-only transport deadline exceeded') from None

    def _fetch(self,pages):
        if not pages:return
        self.poll_exchanges=getattr(self,'poll_exchanges',0)+1
        self._charge(len(pages)*4096)
        def exchange():
            self._process.stdin.write((str(len(pages))+'\n'+''.join(f'{page:x}\n' for page in pages)).encode('ascii'))
            self._process.stdin.flush()
            values={};errors=[]
            for page in pages:
                line=self._line()
                if not line.startswith(b'PAGE '):raise ValueError('Read-only transport framing lost')
                status=int(line.split()[1])
                if status:errors.append((page,status))
                else:
                    data=base64.b64decode(self._line(),validate=True)
                    if len(data)!=4096:raise ValueError('Short read-only page response')
                    values[page]=data
            return values,errors
        values,errors=self._bounded(exchange)
        self._page_cache.update(values)
        if errors:
            address,status=errors[0]
            error_type=UnmappedGamePageError if status in (errno.EIO,errno.EFAULT) else OSError
            raise error_type(status,f'Unmapped or changed game page at {address:#x}')

    def prefetch_pages(self,addrs):
        pages=sorted({(a&_PTR_MASK)&~4095 for a in addrs if is_ptr(a)}-self._page_cache.keys())
        self._fetch(pages)

    def _read_pages(self,start_page,count):
        self._fetch([start_page+i*4096 for i in range(count) if start_page+i*4096 not in self._page_cache])
        return b''.join(self._page_cache[start_page+i*4096] for i in range(count))

    def close(self):
        if getattr(self,'_process',None):
            if self._process.poll() is None:self._process.kill()
            try:self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
            for stream in (self._process.stdin,self._process.stdout):
                if stream:
                    try:stream.close()
                    except OSError:pass
        if getattr(self,'_executor',None):self._executor.shutdown(wait=False,cancel_futures=True)
