import logging
from time import perf_counter_ns

from rich.logging import RichHandler
from rich.traceback import install as rich_traceback_install

handler = RichHandler()
handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.addHandler(handler)


def in_notebook() -> bool:
    """Check if code is executed in Jupyter Notebook"""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        return True
    except ImportError:
        return False


if not in_notebook():
    rich_traceback_install()


def fn_log():
    def decorator(f):
        def inner_func(*__args__, **__kwargs__):
            try:
                logger.info(f"Executing: {f.__name__}")
                start = perf_counter_ns()
                res = f(*__args__, **__kwargs__)
                end = perf_counter_ns()
                duration = format_duration(end - start, nano=True)
                logger.info(f"Successfully executed: {f.__name__} in {duration}")
                return res
            except BaseException:
                # stack_trace = "\n".join(traceback.format_exc().splitlines()[3:])
                # logger.error(f"Error in: {f.__name__}\n{stack_trace}", exc_info=False)
                logger.error(f"Error in: {f.__name__}")
                raise

        return inner_func

    return decorator


def format_duration(duration: float, nano: bool = False) -> str:
    if nano:
        duration /= 1e9
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:.0f}h:{minutes:.0f}m:{seconds:.2f}s"
