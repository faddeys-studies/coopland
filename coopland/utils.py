import signal
import contextlib
import yaml
import dacite
from typing import TypeVar, Type


T = TypeVar("T")


@contextlib.contextmanager
def interrupt_atomic(n_interrupts_to_catch=None):
    def handler(*args, **kwargs):
        nonlocal n_catched
        n_catched += 1
        if n_interrupts_to_catch is not None:
            if n_catched > n_interrupts_to_catch:
                raise KeyboardInterrupt
        print("Will interrupt after atomic operation")
        nonlocal interrupted
        interrupted = True

    interrupted = False
    n_catched = 0

    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    yield
    signal.signal(signal.SIGINT, prev_handler)
    if interrupted:
        raise KeyboardInterrupt


def load_from_yml(datatype: Type[T], filename) -> T:
    with open(filename) as f:
        data = yaml.safe_load(f)
    return dacite.from_dict(datatype, data)
