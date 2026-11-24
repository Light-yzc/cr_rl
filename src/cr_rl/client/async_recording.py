"""Bounded, ordered forensic I/O off the real-time inference thread."""
import queue
import threading


class OrderedRecorder:
    def __init__(self, capacity=64):
        self.queue=queue.Queue(maxsize=capacity);self.error=None;self.closed=False
        self.thread=threading.Thread(target=self._run,name='nulls-recorder',daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            task=self.queue.get()
            try:
                if task is None:return
                if self.error is None:
                    function,args,kwargs=task
                    try:function(*args,**kwargs)
                    except BaseException as error:self.error=error
            finally:self.queue.task_done()

    def submit(self,function,*args,**kwargs):
        if self.closed:raise RuntimeError('Recorder is closed')
        if self.error is not None:raise RuntimeError('Background recording failed') from self.error
        try:self.queue.put_nowait((function,args,kwargs))
        except queue.Full as error:
            raise RuntimeError('Recording backlog exceeded bound; stopping without dropping frames') from error

    def flush(self):
        self.queue.join()
        if self.error is not None:raise RuntimeError('Background recording failed') from self.error

    def close(self):
        if self.closed:return
        self.queue.put(None);self.thread.join();self.closed=True
        if self.error is not None:raise RuntimeError('Background recording failed') from self.error
