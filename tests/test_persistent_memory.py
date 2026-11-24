import base64
import io
from types import SimpleNamespace

import pytest

from cr_rl.client.nulls_reader import BoundedAdbMemReader
from cr_rl.client.persistent_memory import PersistentAdbMemReader, UnmappedGamePageError


def test_failed_page_drains_entire_batch_before_next_request():
    data=bytes(range(256))*16
    response=b'PAGE 0\n'+base64.b64encode(data)+b'\n'
    backend=PersistentAdbMemReader.__new__(PersistentAdbMemReader)
    BoundedAdbMemReader.__init__(backend,'test','1')
    backend._process=SimpleNamespace(stdin=io.BytesIO(),stdout=io.BytesIO(b'PAGE 5\n'+response+response))
    backend._bounded=lambda operation:operation()
    with pytest.raises(UnmappedGamePageError) as caught:
        backend._fetch([0x24000000000,0x20000000])
    assert caught.value.errno==5
    assert backend._page_cache[0x20000000]==data
    backend.begin_poll()  # A failed snapshot must not reuse its cached pages.
    backend._fetch([0x20000000])
    assert backend._page_cache[0x20000000]==data
    assert backend.poll_bytes_read==4096


def test_close_ignores_windows_broken_pipe_oserror():
    class BrokenStream:
        def close(self):
            raise OSError(22, 'Invalid argument')

    backend = PersistentAdbMemReader.__new__(PersistentAdbMemReader)
    backend._process = SimpleNamespace(
        poll=lambda: 0,
        wait=lambda timeout=2: 0,
        stdin=BrokenStream(),
        stdout=BrokenStream(),
    )
    backend._executor = None
    backend.close()
