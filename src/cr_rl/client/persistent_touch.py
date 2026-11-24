"""Persistent Android InputManager transport for an already calibrated tap pair."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import subprocess


class PersistentAdbTouch:
    def __init__(self, serial, *, adb='adb', gap_ms=20, timeout=3.):
        if not 0<=gap_ms<=200:raise ValueError('Tap gap must be within 0..200ms')
        self.gap_ms,self.timeout=gap_ms,timeout
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='nulls-touch')
        self.process=subprocess.Popen([adb,'-s',serial,'shell','-T',
            'CLASSPATH=/data/local/tmp/crrl-touch.dex app_process /system/bin CrrlTouch'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            if self._bounded(self._line)!='READY':raise RuntimeError('Touch helper unavailable')
        except Exception:self.close();raise

    def _line(self):
        line=self.process.stdout.readline(512)
        if not line.endswith(b'\n'):raise EOFError('Touch transport disconnected')
        return line.decode('ascii').strip()

    def _bounded(self, operation):
        future=self.executor.submit(operation)
        try:return future.result(timeout=self.timeout)
        except TimeoutError:
            self.close();raise TimeoutError('Touch command timed out; never retry partial taps') from None

    def __call__(self, taps):
        if len(taps)!=2 or any(type(v) is not int for xy in taps for v in xy):
            raise ValueError('Exactly two integer-coordinate taps required')
        if any(not (0<=x<900 and 0<=y<1600) for x,y in taps):
            raise ValueError('Tap outside calibrated screen')
        def exchange():
            command='PLAY '+' '.join(str(v) for xy in taps for v in xy)+f' {self.gap_ms}\n'
            self.process.stdin.write(command.encode('ascii'));self.process.stdin.flush()
            response=self._line()
            if not response.startswith('SENT '):raise RuntimeError(f'Touch unconfirmed: {response}')
            return {'android_touch_ms':int(response.split()[1])}
        return self._bounded(exchange)

    def close(self):
        if self.process.poll() is None:self.process.kill()
        try:self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
        for stream in (self.process.stdin,self.process.stdout):
            if stream:stream.close()
        self.executor.shutdown(wait=False,cancel_futures=True)
