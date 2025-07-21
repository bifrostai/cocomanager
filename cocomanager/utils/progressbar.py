from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

from .logger import in_notebook


class ProgressBar:
    """
    Custom progress bar class that uses rich if in a terminal and tqdm if in a notebook.
    """

    def __init__(self, desc: str | None = None, total: int | None = None, keep_alive: bool = True):
        """
        Parameters
        ----------
        desc : str, optional
            Description of the progress bar, by default None
        total : int, optional
            Total number of steps, by default None
        keep_alive : bool, optional
            Whether to keep the progress bar alive after completion, by default True
        """
        self.in_notebook = in_notebook()
        desc = desc or ""

        if self.in_notebook:
            # NOTE: coloring the leaves behind artifacts in the notebook if `leave=False`
            desc = self.__color_tqdm_desc(desc) if keep_alive else desc
            color = "green" if keep_alive else None
            self.tqdm_bar = tqdm(desc=desc, total=total, colour=color, leave=keep_alive)
        else:
            PROG_BAR_COLS = [
                TextColumn("[cyan2]{task.description}[/]"),
                BarColumn(),
                TextColumn("[green]{task.percentage:.3g}%[/]"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                SpinnerColumn(),
            ]
            transient = not keep_alive
            self.p_bar = Progress(*PROG_BAR_COLS, transient=transient)
            if desc is not None:
                self._init_task = self.p_bar.add_task(desc, total=total)
            self.p_bar.start()

    def __color_tqdm_desc(self, desc: str, color_code: str = "36") -> str:
        """
        Colors the description of tqdm progress bar
        """
        return f"\033[{color_code}m{desc}\033[0m"

    def add_task(self, description: str, total: int, **kwargs) -> TaskID | None:
        """
        Adds a task to rich progress bar
        """
        task = None if self.in_notebook else self.p_bar.add_task(description, total=total, **kwargs)
        return task

    def advance(self, value: int = 1, task: TaskID | None = None) -> None:
        """
        Advances the progress bar by the given value
        """
        if self.in_notebook:
            self.tqdm_bar.update(value)
        else:
            if task is None:
                task = self._init_task
            self.p_bar.update(task_id=task, advance=value)
        return

    def update(self, advance: int = 1, task: TaskID | None = None, **kwargs) -> None:
        """
        Updates the progress bar to the given value
        """
        if self.in_notebook:
            self.tqdm_bar.update(advance)
        else:
            if task is None:
                task = self._init_task
            self.p_bar.update(task_id=task, advance=advance, **kwargs)
        return

    def write(self, msg: str) -> None:
        """
        Writes a message to the progress bar
        """
        msg = str(msg)
        self.tqdm_bar.write(msg) if self.in_notebook else self.p_bar.console.print(msg)
        return

    def refresh(self) -> None:
        """
        Force refresh the progress bar
        """
        if self.in_notebook:
            self.tqdm_bar.refresh()
        else:
            self.p_bar.refresh()

    def close(self):
        self.tqdm_bar.close() if self.in_notebook else self.p_bar.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.in_notebook:
            self.tqdm_bar.close()
        self.close()
