import importlib.metadata
from typing import List

from cocomanager.core.functions import annotate_all_images, concat, validatate_coco
from cocomanager.core.parser import COCOParser


def __dir__() -> List[str]:
    return list(globals().keys())


__doc__ = """Package to manage COCO annotations"""
__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    # functions
    "concat",
    "validatate_coco",
    "annotate_all_images",
    # parser
    "COCOParser",
]
