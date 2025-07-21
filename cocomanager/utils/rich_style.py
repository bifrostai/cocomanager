from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

PROG_BAR_COLS = [
    TextColumn("[cyan3]{task.description}[/]"),
    BarColumn(),
    TextColumn("[green]{task.percentage:.3g}%[/]"),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    SpinnerColumn(),
]
