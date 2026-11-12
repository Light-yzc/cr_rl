"""One CPU microbatch ahead; deterministic order and bounded host memory."""
from concurrent.futures import ThreadPoolExecutor


def prefetch_cpu_batches(starts, selected, size, build, pin=False):
    starts = iter(starts)

    def prepare(start):
        tensors = build(selected[start:start + size], target_device='cpu')
        if pin:
            tensors = tuple(t.pin_memory() for t in tensors)
        return tensors

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='bc-batch') as pool:
        current = next(starts, None)
        if current is None:
            return
        pending = pool.submit(prepare, current)
        while True:
            tensors = pending.result()
            following = next(starts, None)
            if following is not None:
                pending = pool.submit(prepare, following)
            yield current, tensors
            if following is None:
                return
            current = following
