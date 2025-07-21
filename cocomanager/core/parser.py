from __future__ import annotations

import math
import os
import shutil
import xml.etree.cElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time
from typing import Any, Iterable, Literal, cast, overload

import imagesize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ujson as json
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
from matplotlib.text import Text
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.io.formats.style import Styler
from PIL import Image

from ..models.parser_methods import UpdateAnnotationModel
from ..utils.logger import fn_log, logger
from ..utils.progressbar import ProgressBar


class COCOParser:
    def __init__(self, fpath: str | None = None, img_dir: str | None = None) -> None:
        self.images: pd.DataFrame = pd.DataFrame()
        self.categories: pd.DataFrame = pd.DataFrame()
        self.annotations: pd.DataFrame = pd.DataFrame()

        self.img_dir = img_dir

        if fpath is not None:
            self.__parse_file(fpath)

    @fn_log()
    def __parse_file(self, fpath: str) -> None:
        json_data = json.load(open(fpath, "r"))

        img_df = pd.DataFrame(json_data["images"])
        if img_df.empty:
            img_df = pd.DataFrame(columns=["id", "file_name", "width", "height"])

        cat_df = pd.DataFrame(json_data["categories"])
        if cat_df.empty:
            cat_df = pd.DataFrame(columns=["id", "name"])

        annotations_df = pd.DataFrame(json_data["annotations"])
        if annotations_df.empty:
            annotations_df = pd.DataFrame(
                columns=["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"]
            )

        self.images = img_df.sort_values("id", ignore_index=True)
        self.categories = cat_df.sort_values("id", ignore_index=True)
        self.annotations = annotations_df.sort_values("id", ignore_index=True)

    @classmethod
    @fn_log()
    def parse_from_voc(cls, directory: str, img_dir: str | None = None) -> COCOParser:
        """
        [Beta] Parse VOC annotations (.xml) to COCO format

        Parameters
        ----------
        directory : str
            Directory containing VOC annotations (.xml)

        img_dir : str, optional
            Base path to image directory with respect to `path` attribute in VOC annotation files

        Returns
        ----------
        COCOParser
        """
        if not Path(directory).is_dir():
            raise FileNotFoundError(f"{directory} is not a directory")
        xml_files = list(Path(directory).glob("./*.xml"))

        # NOTE: multithread if too slow
        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        coco_categories: list[dict] = []
        cat_dict: dict = {}

        with ProgressBar(total=len(xml_files), desc="Reading VOC files", keep_alive=False) as pbar:
            for img_idx, fpath in enumerate(sorted(xml_files), 1):
                tree = ET.parse(fpath)
                root = tree.getroot()

                # images
                filename_elem = root.find("filename")
                if filename_elem is None or filename_elem.text is None:
                    raise ValueError("Missing filename element in XML")

                size_elem = root.find("size")
                if size_elem is None:
                    raise ValueError("Missing size element in XML")

                width_elem = size_elem.find("width")
                height_elem = size_elem.find("height")
                if width_elem is None or width_elem.text is None:
                    raise ValueError("Missing width element in XML")
                if height_elem is None or height_elem.text is None:
                    raise ValueError("Missing height element in XML")

                coco_image_info = {
                    "id": img_idx,
                    "file_name": filename_elem.text.lstrip("/"),
                    "width": int(width_elem.text),
                    "height": int(height_elem.text),
                }
                coco_images.append(coco_image_info)

                # annotations
                for anno in root.findall("object"):
                    bnbbox = anno.find("bndbox")
                    if bnbbox is None:
                        raise ValueError("Missing bounding box element in XML")

                    xmin_elem = bnbbox.find("xmin")
                    ymin_elem = bnbbox.find("ymin")
                    xmax_elem = bnbbox.find("xmax")
                    ymax_elem = bnbbox.find("ymax")

                    if (
                        xmin_elem is None
                        or xmin_elem.text is None
                        or ymin_elem is None
                        or ymin_elem.text is None
                        or xmax_elem is None
                        or xmax_elem.text is None
                        or ymax_elem is None
                        or ymax_elem.text is None
                    ):
                        raise ValueError("Missing bounding box element in XML")

                    xmin = round(float(xmin_elem.text))
                    ymin = round(float(ymin_elem.text))
                    xmax = round(float(xmax_elem.text))
                    ymax = round(float(ymax_elem.text))
                    score = bnbbox.find("score")
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                    # categories
                    name_elem = anno.find("name")
                    if name_elem is None or name_elem.text is None:
                        raise ValueError("Missing name element in XML")
                    cat_name = name_elem.text
                    if cat_name not in cat_dict:
                        cat_idx = len(coco_categories) + 1
                        cat_dict[cat_name] = cat_idx
                        coco_categories.append(
                            {
                                "id": cat_idx,
                                "name": cat_name,
                            }
                        )
                    else:
                        cat_idx = cat_dict[cat_name]

                    coco_annotation_info = {
                        "id": len(coco_annotations) + 1,
                        "image_id": img_idx,
                        "category_id": cat_idx,
                        "bbox": bbox,
                        "area": (xmax - xmin) * (ymax - ymin),
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                    if score is not None and score.text is not None:
                        coco_annotation_info["score"] = float(score.text)
                    coco_annotations.append(coco_annotation_info)

                pbar.advance()
            pbar.refresh()

        new_parser = cls()
        new_parser.images = (
            pd.DataFrame(coco_images) if coco_images else pd.DataFrame(columns=["id", "file_name", "width", "height"])
        )
        new_parser.categories = (
            pd.DataFrame(coco_categories) if coco_categories else pd.DataFrame(columns=["id", "name"])
        )
        new_parser.annotations = (
            pd.DataFrame(coco_annotations)
            if coco_annotations
            else pd.DataFrame(columns=["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"])
        )
        new_parser.img_dir = img_dir
        return new_parser

    @classmethod
    @fn_log()
    def parse_from_yolo(
        cls,
        directory: str,
        img_dir: str,
        category_map: dict[int, str] | str | None = None,
    ) -> COCOParser:
        """
        [Beta] Parse YOLO annotations (.txt) to COCO format

        Parameters
        ----------
        directory : str
            Directory containing YOLO annotations (.txt)

        img_dir : str
            Path to images

        category_map : dict[int, str] | str, optional
            Mapping of category index to category name, or path to data.yaml file

        Returns
        ----------
        COCOParser
        """
        accepted_formats = list(Image.registered_extensions().keys())
        yolo_images = list(filter(lambda f: f.suffix in accepted_formats, Path(img_dir).iterdir()))

        # NOTE: multithread if too slow
        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        cat_idxs: set[int] = set()

        with ProgressBar(total=len(yolo_images), desc="Reading YOLO files", keep_alive=False) as pbar:
            for img_idx, fpath in enumerate(sorted(yolo_images), 1):
                # get image size
                width, height = imagesize.get(fpath)
                if width == height == -1:
                    width, height = Image.open(fpath).size

                # images
                coco_image_info = {
                    "id": img_idx,
                    "file_name": fpath.name,
                    "width": width,
                    "height": height,
                }
                coco_images.append(coco_image_info)

                # annotations
                anno_fpath = Path(directory) / fpath.with_suffix(".txt").name
                if anno_fpath.is_file():
                    annos = open(anno_fpath).readlines()
                    for anno in annos:
                        cat_idx, x_mid_fraction, y_mid_fraction, width_fraction, height_fraction, *score = map(
                            float, anno.split(" ")
                        )
                        cat_idxs.add(int(cat_idx))
                        bbox_width, bbox_height = width_fraction * width, height_fraction * height
                        x_tl = (x_mid_fraction * width) - (bbox_width / 2)
                        y_tl = (y_mid_fraction * height) - (bbox_height / 2)

                        coco_annotation_info = {
                            "id": len(coco_annotations) + 1,
                            "image_id": img_idx,
                            "category_id": int(cat_idx),
                            "bbox": [x_tl, y_tl, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                        if score:
                            coco_annotation_info["score"] = score[0]
                        coco_annotations.append(coco_annotation_info)

                pbar.advance()
            pbar.refresh()

            # categories
            cat_idx_list = sorted(cat_idxs)
            if category_map is None:
                coco_categories = [{"id": idx, "name": str(idx)} for idx in cat_idx_list]
            elif isinstance(category_map, str):
                if Path(category_map).suffix not in [".yaml", ".yml"]:
                    raise ValueError("Invalid category_map file. Must be .yaml or .yml")
                names_attr = yaml.safe_load(open(category_map))["names"]
                if isinstance(names_attr, list):
                    if len(names_attr) != len(cat_idx_list):
                        raise ValueError(
                            "Length of classes in `names` not equal number of unique classes in annotation files. "
                            f"{len(names_attr)} vs {len(cat_idx_list)}"
                        )
                    coco_categories = [{"id": idx, "name": cat} for idx, cat in zip(cat_idx_list, names_attr)]
                elif isinstance(names_attr, dict):
                    if not set(cat_idx_list).issubset(set(names_attr.keys())):
                        raise ValueError(
                            f"Missing category IDs {set(cat_idx_list) - set(names_attr.keys())} in category_map file"
                        )
                    coco_categories = [{"id": idx, "name": cat} for idx, cat in names_attr.items()]
                else:
                    raise ValueError(f"Invalid `names` attribute. Type must be list or dict but got {names_attr}")
            elif isinstance(category_map, dict):
                if not set(cat_idx_list).issubset(set(category_map.keys())):
                    raise ValueError(
                        f"Missing category IDs {set(cat_idx_list) - set(category_map.keys())} in category_map"
                    )
                coco_categories = [{"id": idx, "name": cat} for idx, cat in category_map.items()]
            else:
                raise ValueError(f"Invalid category_map type. Must be dict or str but got {type(category_map)}")

            new_parser = cls()
            new_parser.images = (
                pd.DataFrame(coco_images)
                if coco_images
                else pd.DataFrame(columns=["id", "file_name", "width", "height"])
            )
            new_parser.categories = (
                pd.DataFrame(coco_categories) if coco_categories else pd.DataFrame(columns=["id", "name"])
            )
            new_parser.annotations = (
                pd.DataFrame(coco_annotations)
                if coco_annotations
                else pd.DataFrame(columns=["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"])
            )
            new_parser.img_dir = img_dir
            return new_parser

    @property
    def data(self) -> pd.DataFrame:
        if self.annotations.shape[1] == 0:
            raise AttributeError("Unable to load `data` without a COCO file")

        img_df = self.images.copy().rename(columns={"id": "image_id"})
        cat_df = self.categories.copy().rename(columns={"id": "category_id"})
        df = self.annotations.merge(cat_df, on="category_id", how="left").merge(img_df, on="image_id", how="left")
        df = self.__expand_coco_bbox_info(df)

        return df

    def __expand_coco_bbox_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional bbox information for COCO annotations
        """
        if df.empty:
            return pd.concat(
                [
                    df,
                    pd.DataFrame(
                        columns=[
                            "bbox_x",
                            "bbox_y",
                            "bbox_w",
                            "bbox_h",
                            "bbox_w_p",
                            "bbox_h_p",
                            "bbox_center_x",
                            "bbox_center_y",
                            "bbox_center_x_p",
                            "bbox_center_y_p",
                        ]
                    ),
                ]
            )

        df["bbox_x"] = df["bbox"].str[0]
        df["bbox_y"] = df["bbox"].str[1]
        df["bbox_w"] = df["bbox"].str[2]
        df["bbox_h"] = df["bbox"].str[3]

        df["bbox_w_p"] = df["bbox_w"] / df["width"] * 100
        df["bbox_h_p"] = df["bbox_h"] / df["height"] * 100

        df["bbox_center_x"] = df["bbox"].str[0] + df["bbox"].str[2] / 2
        df["bbox_center_y"] = df["bbox"].str[1] + df["bbox"].str[3] / 2

        df["bbox_center_x_p"] = df["bbox_center_x"] / df["width"] * 100
        df["bbox_center_y_p"] = df["bbox_center_y"] / df["height"] * 100

        return df

    # region: miscellaneous
    def get_categories_mapping(self) -> dict[int, str]:
        """
        Get category ID to category name mapping

        Returns
        ----------
        categories_mapping: Dict[int, str]
        """
        if self.categories.empty:
            return dict()

        cat_array = self.categories[["id", "name"]].values
        return dict((id_, name) for id_, name in cat_array)

    def get_images_mapping(self) -> dict[int, str]:
        """
        Get images ID to image file name mapping

        Returns
        ----------
        images_mapping: Dict[int, str]
        """
        if self.images.empty:
            return dict()

        img_array = self.images[["id", "file_name"]].values
        return dict((id_, name) for id_, name in img_array)

    @overload
    def validate_bbox(self, show_result: Literal[False] = False, style: bool = False) -> bool: ...

    @overload
    def validate_bbox(self, show_result: Literal[True], style: Literal[False] = False) -> pd.DataFrame: ...

    @overload
    def validate_bbox(self, show_result: Literal[True], style: Literal[True]) -> Styler: ...

    def validate_bbox(self, show_result: bool = False, style: bool = False) -> bool | pd.DataFrame | Styler:
        """
        Validates bbox (xywh)

        Checks:
            x1 and y1 >= 0
            x2 <= image width
            y2 <= image height
            bbox height and width > 0

        Parameters
        ----------
        show_result : bool, default=False
            Show result in the form of a pandas dataframe

        style : bool, default=False
            Apply styling to output pandas dataframe (only applicable if show_result=True)

        Returns
        ----------
        bool | pd.DataFrame
            True if no errors, False if errors exist. If show_result=True, returns pandas dataframe of errors

        See Also
        ----------
        clip_bbox: Clip bbox values according to image height and width
        """
        error_df = self.data[["id", "file_name", "width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()

        error_df = error_df.loc[
            (error_df["bbox_x"] < 0)
            | (error_df["bbox_y"] < 0)
            | (error_df["bbox_w"] <= 0)
            | (error_df["bbox_h"] <= 0)
            | (error_df["bbox_x"] + error_df["bbox_w"] > error_df["width"])
            | (error_df["bbox_y"] + error_df["bbox_h"] > error_df["height"])
        ]

        def highlight_negative(x):
            return ["color: red" if v < 0 else "" for v in x]

        def highlight_non_positive(x):
            return ["color: red" if v <= 0 else "" for v in x]

        def highlight_exceeds_image_size(row):
            background_colors = [""] * len(row)
            if row["bbox_x"] + row["bbox_w"] > row["width"]:
                background_colors[4] = "background: yellow"
                background_colors[6] = "background: yellow"
            if row["bbox_y"] + row["bbox_h"] > row["height"]:
                background_colors[5] = "background: yellow"
                background_colors[7] = "background: yellow"
            return background_colors

        if show_result:
            if style:
                styled_df = (
                    error_df.style.apply(highlight_negative, subset=["bbox_x", "bbox_y"])
                    .apply(highlight_non_positive, subset=["bbox_w", "bbox_h"])
                    .apply(highlight_exceeds_image_size, axis=1)
                )
                return styled_df
            return error_df
        return error_df.empty

    def copy(self) -> COCOParser:
        """
        Copy the COCOParser
        """
        new_parser = COCOParser()
        new_parser.images = self.images.copy()
        new_parser.categories = self.categories.copy()
        new_parser.annotations = self.annotations.copy()
        new_parser.img_dir = self.img_dir
        return new_parser

    # endregion

    # region: edit
    @fn_log()
    def match_categories(
        self,
        other: COCOParser | dict[int, str],
        include_extra: bool = False,
        inplace: bool = False,
        match_on: str = "name",
        show_changes: bool = False,
    ) -> COCOParser:
        """
        Match categories mapping to another COCO category mapping

        Parameters
        ----------
        other : COCOParser | dict
            Category mapping to match


        include_extra: bool, default=False
            To include extra categories from the other category mapping

        inplace: bool, default=False
            Perform changes in place

        match_on : {'name', 'id'}
            Match new category base on name or id. Update the other. default=name

        Examples
        ----------
        >>> # after loading parsers
        >>> parser_old.get_categories_mapping()
        {1: 'cat_a', 2: 'cat_b', 3: 'cat_c'}

        >>> parser_new.get_categories_mapping()
        {1: 'cat_a', 2: 'cat_c', 3: 'cat_b', 4: 'cat_d'}

        >>> parser_old.match_categories(parser_new, show_changes=True)
        {'cat_b': {'old': 2, 'new': 3}, 'cat_c': {'old': 3, 'new': 2}}

        >>> parser_old.match_categories(parser_new, include_extra=True, show_changes=True)
        {'cat_b': {'old': 2, 'new': 3},
        'cat_c': {'old': 3, 'new': 2},
        'cat_d': {'old': None, 'new': 4}}
        """
        if isinstance(other, COCOParser):
            other_id2name = other.get_categories_mapping()
        elif isinstance(other, dict):
            other_id2name = other
        else:
            raise TypeError(f"{type(other)} is not a valid input type for `other`")

        if match_on not in ["name", "id"]:
            raise ValueError(f"{match_on} is not a valid input for `match_on`")

        initial_id2name = self.get_categories_mapping()

        if match_on == "name":
            self_cat_name = set(initial_id2name.values())
            other_cat_name = set(other_id2name.values())

            # ensure no duplicate category name
            if len(self_cat_name) != len(initial_id2name):
                raise Exception("Unable to match on `name` as non-unique name(s) found for self")

            if len(other_cat_name) != len(other_id2name):
                raise Exception("Unable to match on `name` as non-unique name(s) found for other")

            # ensure all categories names are present in new categories
            missing_cat_name = self_cat_name - other_cat_name
            if missing_cat_name:
                raise Exception(f"Missing names {list(missing_cat_name)} in new categories")

            # update copy of category dataframe
            tmp_cat_df = self.categories.copy()
            other_mapping_flip = dict((v, k) for k, v in other_id2name.items())
            tmp_cat_df["id_new"] = tmp_cat_df["name"].map(other_mapping_flip)
            if include_extra:
                new_cat_names = other_cat_name - self_cat_name
                tmp_cat_df = pd.concat(
                    [
                        tmp_cat_df,
                        pd.DataFrame(
                            [
                                {
                                    "id": None,
                                    "name": new_cat_name,
                                    "id_new": other_mapping_flip[new_cat_name],
                                }
                                for new_cat_name in new_cat_names
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            if show_changes:
                change_dict = {}
                for name, old_id, new_id in tmp_cat_df[["name", "id", "id_new"]].values:
                    if old_id != new_id:
                        change_dict[name] = {"old": old_id, "new": new_id}
                print(change_dict)

            cat_array = tmp_cat_df[["id", "id_new"]].values
            new_cat_mapping = dict((old, new) for old, new in cat_array)

            new_cat_df = tmp_cat_df.copy()
            new_cat_df["id"] = new_cat_df["id_new"]
            new_cat_df.drop("id_new", axis=1, inplace=True)
            new_cat_df = new_cat_df.sort_values("id", ignore_index=True)

            new_anno_df = self.annotations.copy()
            new_anno_df["category_id"] = self.annotations["category_id"].map(new_cat_mapping)

            if inplace:
                # update categories
                self.categories = new_cat_df
                # update annotations
                self.annotations = new_anno_df
                return self
            else:
                new_parser = COCOParser()
                new_parser.images = self.images
                new_parser.categories = new_cat_df
                new_parser.annotations = new_anno_df
                new_parser.img_dir = self.img_dir
                return new_parser

        else:
            self_cat_ids = set(initial_id2name.keys())
            other_cat_ids = set(other_id2name.keys())

            # ensure all categories ids are present in new categories
            missing_cat_ids = self_cat_ids - other_cat_ids
            if missing_cat_ids:
                raise Exception(f"Missing IDs {list(missing_cat_ids)} in new categories")

            # update copy of category dataframe
            tmp_cat_df = self.categories.copy()
            tmp_cat_df["name_new"] = tmp_cat_df["id"].map(other_id2name)
            if include_extra:
                new_cat_ids = other_cat_ids - self_cat_ids
                tmp_cat_df = pd.concat(
                    [
                        tmp_cat_df,
                        pd.DataFrame(
                            [
                                {"id": new_cat_id, "name": None, "name_new": other_id2name[new_cat_id]}
                                for new_cat_id in new_cat_ids
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            if show_changes:
                change_dict = {}
                for id_, old_cat, new_cat in tmp_cat_df[["id", "name", "name_new"]].values:
                    if old_cat != new_cat:
                        change_dict[id_] = {"old": old_cat, "new": new_cat}
                print(change_dict)

            new_cat_df = tmp_cat_df.copy()
            new_cat_df["name"] = new_cat_df["name_new"]
            new_cat_df.drop("name_new", axis=1, inplace=True)
            new_cat_df = new_cat_df.sort_values("id", ignore_index=True)

            if inplace:
                # update categories
                self.categories = new_cat_df
                return self
            else:
                new_parser = COCOParser()
                new_parser.images = self.images
                new_parser.categories = new_cat_df
                new_parser.annotations = self.annotations
                new_parser.img_dir = self.img_dir
                return new_parser

    @fn_log()
    def rename_categories(
        self,
        mapping: dict[str, str],
        reset_id: bool = False,
        inplace: bool = False,
    ) -> COCOParser:
        """
        Rename categories with a dictionary
        Names not in the input dictionary but in the COCO categories will remain unchanged.
        Names in the input dictionary but not in the COCO categories will be ignored.

        Parameters
        ----------
        mapping : Dictionary[str, str]
            Mapping to change the categories' name

        reset_id : bool, default=False
            Reset ID to be sequential

        inplace : bool, default=False
            Perform changes in place

        Examples
        ----------
        >>> # after loading parsers
        >>> parser.get_categories_mapping()
        {1: 'cat_a', 2: 'cat_b', 3: 'cat_c'}

        >>> parser.rename_categories({"cat_a": "cat_x", "cat_b":"cat_x", "extra": "extra_new"}, reset_index=True, inplace=True)

        >>> parser.get_categories_mapping()
        {1: 'cat_x', 2: 'cat_c'}

        """
        # Ensure new name is of type `string`
        if any(not isinstance(_, str) for _ in mapping.values()):
            raise Exception("New category names have to be of type Str")

        tmp_cat_df = self.categories.copy()
        # replace category names
        tmp_cat_df["name_new"] = tmp_cat_df["name"].replace(mapping)

        if reset_id:
            tmp_cat_df["id_new"] = pd.factorize(tmp_cat_df["name_new"])[0] + 1
        else:
            tmp_cat_df["id_new"] = tmp_cat_df.groupby("name_new")["id"].transform("min")

        # update annotations
        new_id_map = dict((old, new) for old, new in tmp_cat_df[["id", "id_new"]].values)
        new_annotations_df = self.annotations.copy()
        new_annotations_df["category_id"] = new_annotations_df["category_id"].map(new_id_map)

        # update categories
        new_cat_df = tmp_cat_df.drop_duplicates(subset=["id_new"], keep="first")[["id_new", "name_new"]].copy()
        new_cat_df = new_cat_df.rename(columns={"id_new": "id", "name_new": "name"})
        new_cat_df = new_cat_df.sort_values("id", ignore_index=True)

        if inplace:
            self.annotations = new_annotations_df
            self.categories = new_cat_df
            return self
        else:
            new_parser = COCOParser()
            new_parser.images = self.images
            new_parser.categories = new_cat_df
            new_parser.annotations = new_annotations_df
            new_parser.img_dir = self.img_dir
            return new_parser

    @fn_log()
    def update_annotations(self, updates: dict[int, dict[str, Any]], inplace: bool = False) -> COCOParser:
        """
        Update annotation(s)
        Supported changes:
            - bbox
            - category

        Parameters
        ----------
        updates : Dict[int, Dict[str, Any]]
            Mapping of annotation ID to changes.
            Changes is a mapping of the attribute of change to the new values.

        inplace : bool, default=False
            Perform changes in place

        Examples
        ----------
        >>> # initial annotations
        >>> print(parser.annotations)
            id  image_id  category_id          bbox   area segmentation  iscrowd
        0   1         1            1  [1, 1, 1, 1]  16110           []        0
        1   2         2            2  [2, 2, 2, 2]  16110           []        0
        2   3         3            3  [3, 3, 3, 3]  16110           []        0
        >>> parser.update_annotations({1: {"bbox": [10,20,30,40], "category_id": 2}}).annotations
            id  image_id  category_id              bbox   area segmentation  iscrowd
        0   1         1            2  [10, 20, 30, 40]  16110           []        0
        1   2         2            2      [2, 2, 2, 2]  16110           []        0
        2   3         3            3      [3, 3, 3, 3]  16110           []        0
        """
        anno_ids = set(updates.keys())
        possible_anno_ids = set(self.annotations["id"])
        if not anno_ids.issubset(possible_anno_ids):
            diff = anno_ids - possible_anno_ids
            raise ValueError(f"Annotations IDs not found: {diff}")

        data_list = []
        for id_, update in updates.items():
            # validate update
            update["possible_category_ids"] = self.categories["id"].values.tolist()  # type: ignore
            UpdateAnnotationModel(**update)

            update.pop("possible_category_ids")
            data_list.append({"id": id_, **update})

        # form df of updates
        new_anno_df = pd.DataFrame(data_list)
        # add IDs that are not updated
        anno_copy = self.annotations[["id"]].copy()
        new_anno_df = anno_copy.merge(new_anno_df, on="id", how="left")
        # add back all other annotations that were not changed
        new_anno_df = new_anno_df.combine_first(self.annotations.copy())
        # reorder columns
        new_anno_df = new_anno_df[self.annotations.columns]
        # update column types
        new_anno_df = new_anno_df.astype(self.annotations.dtypes.to_dict())

        if inplace:
            self.annotations = new_anno_df
            return self
        else:
            new_parser = COCOParser()
            new_parser.images = self.images
            new_parser.categories = self.categories
            new_parser.annotations = new_anno_df
            new_parser.img_dir = self.img_dir
            return new_parser

    def clip_bbox(self, inplace: bool = False) -> COCOParser:
        """
        Clips annotations bbox according to image height and width.
        Only bbox values will be changed, area and segmentation will not be changed.
        NOTE: Might run into precision issues if bbox values are floats

        Parameters
        ----------
        inplace : bool, default=False
            Modify the current parser object

        Returns
        ----------
        COCOParser

        See Also
        ----------
        validate_bbox: Validate bbox values
        """
        corrected_df = self.data[["id", "file_name", "width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()

        # get bottom right corner
        corrected_df["bbox_x2"] = corrected_df["bbox_x"] + corrected_df["bbox_w"]
        corrected_df["bbox_y2"] = corrected_df["bbox_y"] + corrected_df["bbox_h"]

        # clip
        corrected_df["bbox_x"] = corrected_df["bbox_x"].clip(0, corrected_df["width"])
        corrected_df["bbox_x2"] = corrected_df["bbox_x2"].clip(0, corrected_df["width"])
        corrected_df["bbox_y"] = corrected_df["bbox_y"].clip(0, corrected_df["height"])
        corrected_df["bbox_y2"] = corrected_df["bbox_y2"].clip(0, corrected_df["height"])

        # update bbox_w and bbox_h
        corrected_df["bbox_w"] = corrected_df["bbox_x2"] - corrected_df["bbox_x"]
        corrected_df["bbox_h"] = corrected_df["bbox_y2"] - corrected_df["bbox_y"]

        # corrected bbox
        corrected_df["bbox_new"] = corrected_df[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values.tolist()

        if inplace:
            self.annotations["bbox"] = corrected_df["bbox_new"]
            return self

        else:
            new_parser = self.copy()
            new_parser.annotations["bbox"] = corrected_df["bbox_new"]
            return new_parser

    @fn_log()
    def __remove_and_filter(
        self,
        tmp_img_df: pd.DataFrame,
        tmp_cat_df: pd.DataFrame,
        tmp_anno_df: pd.DataFrame,
        reset_id: bool,
        inplace: bool,
        drop_empty_images: bool,
    ) -> COCOParser:
        """
        Helper function for remove and filter methods
        """
        if drop_empty_images:
            img_id_w_annotations = tmp_anno_df["image_id"].unique()
            tmp_img_df = tmp_img_df[tmp_img_df["id"].isin(img_id_w_annotations)].copy()

        if reset_id:
            # update image IDs
            tmp_img_df["old_id"] = tmp_img_df["id"].values
            tmp_img_df["id"] = range(1, len(tmp_img_df) + 1)
            img_id_map = tmp_img_df[["old_id", "id"]].set_index("old_id").to_dict()["id"]
            tmp_anno_df["image_id"] = tmp_anno_df["image_id"].map(img_id_map)
            tmp_img_df.drop(columns=["old_id"], inplace=True)

            # update category IDs
            tmp_cat_df["old_id"] = tmp_cat_df["id"]
            tmp_cat_df["id"] = range(1, len(tmp_cat_df) + 1)
            cat_id_map = tmp_cat_df[["old_id", "id"]].set_index("old_id").to_dict()["id"]
            tmp_anno_df["category_id"] = tmp_anno_df["category_id"].map(cat_id_map)
            tmp_cat_df.drop(columns=["old_id"], inplace=True)

            # update annotation IDs
            tmp_anno_df["id"] = range(1, len(tmp_anno_df) + 1)

        if inplace:
            self.images = tmp_img_df.reset_index(drop=True)
            self.categories = tmp_cat_df.reset_index(drop=True)
            self.annotations = tmp_anno_df.reset_index(drop=True)
            return self
        else:
            new_parser = COCOParser()
            new_parser.images = tmp_img_df.reset_index(drop=True)
            new_parser.categories = tmp_cat_df.reset_index(drop=True)
            new_parser.annotations = tmp_anno_df.reset_index(drop=True)
            new_parser.img_dir = self.img_dir
            return new_parser

    @fn_log()
    def remove_annotations(
        self,
        annotation_ids: list[int],
        reset_id: bool = False,
        inplace: bool = False,
        drop_empty_images: bool = False,
    ) -> COCOParser:
        """
        Remove annotations

        Parameters
        ----------
        annotation_ids : list[int]
            IDs to remove

        reset_id : bool, default=False
            Reset image IDs, category IDs and annotations IDs to be sequential, starting from 1

        inplace : bool, default=False
            Perform changes in place

        drop_empty_images : bool, default=False
            Drop images that have no annotations left

        See Also
        --------
        filter_annotations
        remove_images
        remove_categories
        """
        new_annotations = self.annotations[~self.annotations["id"].isin(annotation_ids)].copy()

        return self.__remove_and_filter(
            self.images.copy(), self.categories.copy(), new_annotations, reset_id, inplace, drop_empty_images
        )

    @fn_log()
    def filter_annotations(
        self,
        annotation_ids: list[int],
        reset_id: bool = False,
        inplace: bool = False,
        drop_empty_images: bool = False,
    ) -> COCOParser:
        """
        Filter annotations to keep

        Parameters
        ----------
        annotation_ids : list[int]
            IDs to keep

        reset_id : bool, default=False
            Reset image IDs, category IDs and annotations IDs to be sequential, starting from 1

        inplace : bool, default=False
            Perform changes in place

        drop_empty_images : bool, default=False
            Drop images that have no annotations left

        See Also
        --------
        remove_annotations
        filter_images
        filter_categories
        """
        new_annotations = self.annotations[self.annotations["id"].isin(annotation_ids)].copy()

        return self.__remove_and_filter(
            self.images.copy(), self.categories.copy(), new_annotations, reset_id, inplace, drop_empty_images
        )

    @fn_log()
    def remove_images(
        self,
        images: list[int | str],
        reset_id: bool = False,
        inplace: bool = False,
    ) -> COCOParser:
        """
        Remove images. Annotations of removed images will be deleted too.

        Parameters
        ----------
        img_ids : list[int | str]
            IDs or file names to remove

        reset_id : bool, default=False
            Reset image IDs, category IDs and annotations IDs to be sequential, starting from 1

        inplace : bool, default=False
            Perform changes in place

        See Also
        --------
        filter_images
        remove_annotations
        remove_categories
        """
        removed_img_ids = self.images[self.images["id"].isin(images) | self.images["file_name"].isin(images)][
            "id"
        ].values
        new_images = self.images[~self.images["id"].isin(removed_img_ids)].copy()
        new_annotations = self.annotations[~self.annotations["image_id"].isin(removed_img_ids)].copy()

        return self.__remove_and_filter(new_images, self.categories.copy(), new_annotations, reset_id, inplace, False)

    @fn_log()
    def filter_images(self, images: list[int | str], reset_id: bool = False, inplace: bool = False) -> COCOParser:
        """
        Filter images to keep. Annotations of removed images will be deleted too.

        Parameters
        ----------
        images : list[int | str]
            IDs or file names to keep

        reset_id : bool, default=False
            Reset image and annotations ID to be sequential

        inplace : bool, default=False
            Perform changes in place

        See Also
        --------
        remove_images
        filter_annotations
        filter_categories
        """
        filtered_img_ids = self.images[self.images["id"].isin(images) | self.images["file_name"].isin(images)][
            "id"
        ].values
        new_images = self.images[self.images["id"].isin(filtered_img_ids)].copy()
        new_annotations = self.annotations[self.annotations["image_id"].isin(filtered_img_ids)].copy()

        return self.__remove_and_filter(new_images, self.categories.copy(), new_annotations, reset_id, inplace, False)

    @fn_log()
    def remove_categories(
        self,
        categories: list[int | str],
        reset_id: bool = False,
        inplace: bool = False,
        drop_empty_images: bool = False,
    ) -> COCOParser:
        """
        Remove categories. Annotations of removed categories will be deleted too.

        Parameters
        ----------
        categories : list[int | str]
            IDs or category names to remove

        reset_id : bool, default=False
            Reset image IDs, category IDs and annotations IDs to be sequential, starting from 1

        inplace : bool, default=False
            Perform changes in place

        drop_empty_images : bool, default=False
            Drop images that have no annotations left

        See Also
        --------
        filter_categories
        remove_annotations
        remove_images
        """
        removed_cat_ids = self.categories[
            self.categories["id"].isin(categories) | self.categories["name"].isin(categories)
        ]["id"].values
        new_cat = self.categories[~self.categories["id"].isin(removed_cat_ids)].copy()
        new_annotations = self.annotations[~self.annotations["category_id"].isin(removed_cat_ids)].copy()

        return self.__remove_and_filter(
            self.images.copy(), new_cat, new_annotations, reset_id, inplace, drop_empty_images
        )

    @fn_log()
    def filter_categories(
        self,
        categories: list[int | str],
        reset_id: bool = False,
        inplace: bool = False,
        drop_empty_images: bool = False,
    ) -> COCOParser:
        """
        Filter categories to keep. Annotations of removed categories will be deleted too.

        Parameters
        ----------
        categories : list[int | str]
            IDs or category names to keep

        reset_id : bool, default=False
            Reset image IDs, category IDs and annotations IDs to be sequential, starting from 1

        inplace : bool, default=False
            Perform changes in place

        drop_empty_images : bool, default=False
            Drop images that have no annotations left

        See Also
        --------
        remove_categories
        filter_annotations
        filter_images
        """
        filtered_cat_ids = self.categories[
            self.categories["id"].isin(categories) | self.categories["name"].isin(categories)
        ]["id"].values
        new_cat = self.categories[self.categories["id"].isin(filtered_cat_ids)].copy()
        new_annotations = self.annotations[self.annotations["category_id"].isin(filtered_cat_ids)].copy()

        return self.__remove_and_filter(
            self.images.copy(), new_cat, new_annotations, reset_id, inplace, drop_empty_images
        )

    # endregion

    # region: plot
    def __assert_img_dir(self) -> None:
        """Raise exception if no image directory is provided"""
        if self.img_dir is None:
            raise Exception("img_dir required for plotting")

    def __assert_attr_not_empty(self, attr: str) -> None:
        """Raise exception if attribute of dataframe type is empty"""
        attr_df = getattr(self, attr)
        assert isinstance(attr_df, pd.DataFrame), f"{attr} attribute is not a pandas dataframe but a {type(attr_df)}"
        if attr_df.empty:
            raise Exception(f"No {attr} in COCO file")

    def __generate_ax_patches(
        self,
        df: pd.DataFrame,
        seg: bool,
        bbox: bool,
        text_format: str | None,
        colors: str | list | dict | None,
        seg_kw: dict | None,
        bbox_kw: dict | None,
        text_kw: dict | None,
    ) -> tuple[list[PatchCollection], list[Text]]:
        """
        Helper function to add annotations on image
        """
        seg_polygons: list[Polygon] = []
        bbox_polygons: list[Polygon] = []
        seg_colors: list[str] = []
        bbox_colors: list[str] = []
        collection: list[PatchCollection] = []
        texts: list[Text] = []

        colormap_df = self.categories.copy()
        np.random.seed(5)
        colormap_df["color"] = (np.random.random((len(colormap_df), 3)) * 0.6 + 0.4).tolist()
        if colors is None:
            pass
        elif isinstance(colors, str):
            colormap_df["color"] = colors
        elif isinstance(colors, list):
            colormap_df["color"] = [colors] * len(colormap_df)
        else:
            # allow use of ID or name
            colormap_df["color"] = colormap_df["id"].map(colors).fillna(colormap_df["color"])
            colormap_df["color"] = colormap_df["name"].map(colors).fillna(colormap_df["color"])

        for anno in df.itertuples():
            # select color
            c = colormap_df[colormap_df["id"] == anno.category_id]["color"].values[0]

            # segmentation
            if seg and hasattr(anno, "segmentation"):
                anno_segs = cast(list[list[int]], anno.segmentation)
                if not isinstance(anno_segs, list):
                    raise NotImplementedError()

                for anno_seg in anno_segs:
                    seg_poly = np.array(anno_seg).reshape((int(len(anno_seg) / 2), 2))
                    seg_polygons.append(Polygon(seg_poly.tolist()))
                    seg_colors.append(c)

            # bbox
            anno_bbox = cast(list[float], anno.bbox)
            x, y, w, h = anno_bbox
            if bbox:
                rect_poly = [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                bbox_polygons.append(Polygon(rect_poly))  # type: ignore
                bbox_colors.append(c)

            # text
            text_kw = dict() if text_kw is None else text_kw
            if text_format is not None:
                t = Text(
                    x,
                    y - 1,
                    eval(f"f{repr(text_format)}"),
                    color=text_kw.get("text_color", "black"),
                    bbox=dict(facecolor=c, alpha=text_kw.get("bg_alpha", 0.5), linewidth=0, pad=0.1),
                    wrap=True,
                    clip_on=True,
                    fontproperties=FontProperties(size=text_kw.get("fontsize", 10)),
                )
                t._get_wrap_line_width = lambda: text_kw.get("line_wrap_length", 100)  # type: ignore
                texts.append(t)

        seg_kw = dict() if seg_kw is None else seg_kw
        bbox_kw = dict() if bbox_kw is None else bbox_kw
        # segmentation pixel
        collection.append(
            PatchCollection(seg_polygons, facecolor=seg_colors, linewidths=0, alpha=seg_kw.get("alpha", 0.4))
        )
        # segmentation outline
        collection.append(
            PatchCollection(
                seg_polygons, facecolor="none", edgecolors=seg_colors, linewidths=seg_kw.get("linewidth", 2)
            )
        )
        # bbox outline
        collection.append(
            PatchCollection(
                bbox_polygons, facecolor="none", edgecolors=bbox_colors, linewidths=bbox_kw.get("linewidth", 1.5)
            )
        )

        # reset random seed
        np.random.seed(int(time()))

        return collection, texts

    def __plot(
        self,
        groups: DataFrameGroupBy,
        ncols: int,
        figsize: Iterable[int] | None,
        plot_segmentation: bool,
        plot_bbox: bool,
        text_format: str | None,
        colors: str | list | dict | None,
        title: bool,
        save_path: str | None,
        show: bool,
        seg_kw: dict | None,
        bbox_kw: dict | None,
        text_kw: dict | None,
        **subplot_kw,
    ) -> None:
        """
        Each dataframe in pandas DataFrameGroupBy instance will be a plot
        """
        assert self.img_dir is not None
        # configure MPL
        num_plots = len(groups)
        ncols = min(ncols, num_plots)
        nrows = math.ceil(num_plots / ncols)
        figure, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **subplot_kw)
        plt.setp(axes, xticks=[], yticks=[])  # type: ignore
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        plt.margins(0, 0)
        axs: list[plt.Axes] = axes.flatten()  # type: ignore

        # hide unused plots
        diff = len(axs) - num_plots + 1
        for i in range(1, diff):
            axs[-i].set_visible(False)

        # default text format
        if text_format is None:
            text_format = "{anno.name} {anno.score}" if "score" in self.annotations.columns else "{anno.name}"

        # plot
        ax: plt.Axes
        df: pd.DataFrame
        for ax, (_, df) in zip(axs, groups):
            img_fname: str = df["file_name"].values[0]
            img = plt.imread(os.path.join(self.img_dir, img_fname))

            if not df["bbox"].isnull().all():
                collections, texts = self.__generate_ax_patches(
                    df, plot_segmentation, plot_bbox, text_format, colors, seg_kw, bbox_kw, text_kw
                )
                for c in collections:
                    ax.add_collection(c)  # type: ignore
                for t in texts:
                    ax.add_artist(t)

            if title:
                ax.set_title(img_fname)
            ax.imshow(img)

            # Set axes limits to image dimensions to ensure proper clipping
            height, width = img.shape[:2]
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

        # save
        if save_path is not None:
            figure.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=subplot_kw.get("dpi", "figure"))

        if show:
            plt.show()
        else:
            plt.close()

        return

    def sample_images(
        self,
        sample_size: int = 5,
        seed: int | None = None,
        figsize: Iterable[int] | None = None,
        ncols: int = 1,
        title: bool = True,
        plot_segmentation: bool = False,
        plot_bbox: bool = True,
        text_format: str | None = None,
        colors: str | list | dict | None = None,
        save_path: str | None = None,
        show: bool = True,
        seg_kw: dict | None = None,
        bbox_kw: dict | None = None,
        text_kw: dict | None = None,
        **subplot_kw,
    ) -> None:
        """
        Randomly select images to preview with all bounding boxes and annotations

        Parameters
        ----------
        sample_size : int, default=5
            Number of images to preview

        seed : int, optional
            Seed for sampling

        figsize: Iterable[int] of size 2, optional
            Size of matplotlib figure

        ncols : int, default=1
            Number of columns to arrange plot

        title : bool, default=True
            Display image filename as plot title

        plot_segmentation : bool, default=False
            Plot segmentation mask

        plot_bbox : bool, default=True
            Plot bounding box

        text_format : str, optional, default='{anno.name}'
            Format text above bounding box. Access annotation attribute with `anno`

        colors : str, list, dict, optional
            Color of bbox and segmentation mask.
            str and list input will change all to a single color e.g. 'red', [0.5,0.5,0.5]
            dictionary input will set a specific category to a color e.g. {1: "red", 2: [0.5,0.5,0.5]}

        save_path : str, optional
            Path to save figure

        show : bool, default=True
            Plot figure

        seg_kw : dict, optional
            Segmentation annotation plot settings. `alpha`, `linewidth`

        bbox_kw : dict, optional
            Bounding box annotation plot settings. `linewidth`

        text_kw : dict, optional
            Text annotation plot settings. `bbox_alpha`, `text_color`, `fontsize`

        See Also
        --------
        plot_images: plot specific image(s)
        sample_categories: sample categories to plot
        plot_annotations: plot specific annotation(s)
        """
        self.__assert_img_dir()
        self.__assert_attr_not_empty("images")

        num_plots = max(sample_size, 1)
        num_plots = min(len(self.images), num_plots)
        num_plots = int(num_plots)
        sample_imgs_id = self.images["id"].sample(num_plots, random_state=seed).values

        tmp_data = self.data.loc[self.data["image_id"].isin(sample_imgs_id)]

        # include images without annotations, as it will not appear in tmp_data
        imgs_without_anno = set(sample_imgs_id) - set(tmp_data["image_id"].unique())
        df_to_append = pd.DataFrame(
            [{"image_id": img_id, "file_name": self.get_images_mapping()[img_id]} for img_id in imgs_without_anno]
        )
        tmp_data = pd.concat([tmp_data, df_to_append], ignore_index=True)

        groups = tmp_data.groupby("image_id")

        self.__plot(
            groups,
            ncols,
            figsize,
            plot_segmentation,
            plot_bbox,
            text_format,
            colors,
            title,
            save_path,
            show,
            seg_kw,
            bbox_kw,
            text_kw,
            **subplot_kw,
        )
        return

    def plot_images(
        self,
        img: str | int | Iterable[int | str],
        figsize: Iterable[int] | None = None,
        ncols: int = 1,
        title: bool = True,
        plot_segmentation: bool = False,
        plot_bbox: bool = True,
        text_format: str | None = None,
        colors: str | list | dict | None = None,
        save_path: str | None = None,
        show: bool = True,
        seg_kw: dict | None = None,
        bbox_kw: dict | None = None,
        text_kw: dict | None = None,
        **subplot_kw,
    ) -> None:
        """
        Plot image(s) with all bounding boxes and annotations

        Parameters
        ----------
        img: str or int or Iterable[int | str]
            Image(s) to plot, either an integer representing ID or string representing the file name

        figsize: Iterable[int] of size 2, optional
            Size of matplotlib figure

        ncols : int, default=1
            Number of columns to arrange plot

        title : bool, default=True
            Display image filename as plot title

        plot_segmentation : bool, default=False
            Plot segmentation mask

        plot_bbox : bool, default=True
            Plot bounding box

        text_format : str, optional, default='{anno.name}'
            Format text above bounding box. Access annotation attribute with `anno`

        colors : str, list, dict, optional
            Color of bbox and segmentation mask.
            str and list input will change all to a single color e.g. 'red', [0.5,0.5,0.5]
            dictionary input will set a specific category to a color e.g. {1: "red", 2: [0.5,0.5,0.5]}

        save_path : str, optional
            Path to save figure

        show : bool, default=True
            Plot figure

        seg_kw : dict, optional
            Segmentation annotation plot settings. `alpha`, `linewidth`

        bbox_kw : dict, optional
            Bounding box annotation plot settings. `linewidth`

        text_kw : dict, optional
            Text annotation plot settings. `bbox_alpha`, `text_color`, `fontsize`

        See Also
        --------
        sample_images: sample random images to plot
        sample_categories: sample categories to plot
        plot_annotations: plot specific annotation(s)
        """
        self.__assert_img_dir()
        self.__assert_attr_not_empty("images")

        if isinstance(img, int):
            cond = self.images["id"] == img
        elif isinstance(img, str):
            cond = self.images["file_name"] == img
        elif isinstance(img, (list, tuple)):
            cond = self.images["id"].isin(img) | self.images["file_name"].isin(img)
        else:
            raise TypeError(f"{img} of type {type(img)} is not a valid type")

        sample_imgs_id = self.images[cond]["id"].unique()
        if not len(sample_imgs_id):
            raise ValueError("No image to plot")

        tmp_data = self.data.loc[self.data["image_id"].isin(sample_imgs_id)]

        # include images without annotations, as it will not appear in tmp_data
        imgs_without_anno = set(sample_imgs_id) - set(tmp_data["image_id"].unique())
        df_to_append = pd.DataFrame(
            [{"image_id": img_id, "file_name": self.get_images_mapping()[img_id]} for img_id in imgs_without_anno]
        )
        tmp_data = pd.concat([tmp_data, df_to_append], ignore_index=True)

        groups = tmp_data.groupby("image_id")

        self.__plot(
            groups,
            ncols,
            figsize,
            plot_segmentation,
            plot_bbox,
            text_format,
            colors,
            title,
            save_path,
            show,
            seg_kw,
            bbox_kw,
            text_kw,
            **subplot_kw,
        )
        return

    def sample_categories(
        self,
        category: str | int | Iterable[int | str],
        sample_size: int = 5,
        seed: int | None = None,
        figsize: Iterable[int] | None = None,
        ncols: int = 1,
        title: bool = True,
        plot_segmentation: bool = False,
        plot_bbox: bool = True,
        text_format: str | None = None,
        colors: str | list | dict | None = None,
        save_path: str | None = None,
        show: bool = True,
        seg_kw: dict | None = None,
        bbox_kw: dict | None = None,
        text_kw: dict | None = None,
        **subplot_kw,
    ) -> None:
        """
        Select random samples of categories to preview

        Parameters
        ----------
        category: str or int or Iterable[int | str]
            Category to plot, either an integer representing ID or string representing the name

        sample_size: int, default=5
            Number of images to preview.

        seed: int, optional
            Seed for sampling

        figsize: Iterable[int] of size 2, optional
            Size of matplotlib figure

        ncols : int, default=1
            Number of columns to arrange plot

        title : bool, default=True
            Display image filename as plot title

        plot_segmentation : bool, default=False
            Plot segmentation mask

        plot_bbox : bool, default=True
            Plot bounding box

        text_format : str, optional, default='{anno.name}'
            Format text above bounding box. Access annotation attribute with `anno`

        colors : str, list, dict, optional
            Color of bbox and segmentation mask.
            str and list input will change all to a single color e.g. 'red', [0.5,0.5,0.5]
            dictionary input will set a specific category to a color e.g. {1: "red", 2: [0.5,0.5,0.5]}

        save_path : str, optional
            Path to save figure

        show : bool, default=True
            Plot figure

        seg_kw : dict, optional
            Segmentation annotation plot settings. `alpha`, `linewidth`

        bbox_kw : dict, optional
            Bounding box annotation plot settings. `linewidth`

        text_kw : dict, optional
            Text annotation plot settings. `bbox_alpha`, `text_color`, `fontsize`

        See Also
        --------
        sample_images: sample random images to plot
        plot_images: plot specific image(s)
        plot_annotations: plot specific annotation(s)
        """
        self.__assert_img_dir()
        self.__assert_attr_not_empty("images")

        if isinstance(category, int):
            cond = self.data["category_id"] == category
        elif isinstance(category, str):
            cond = self.data["name"] == category
        elif isinstance(category, (list, tuple)):
            cond = self.data["category_id"].isin(category) | self.data["name"].isin(category)
        else:
            raise TypeError(f"{category} of type {type(category)} is not a valid type")

        sample_size = max(sample_size, 1)
        # find all annotations of specific category
        filter_df = self.data.loc[cond]
        # reduce sample size to total number of images if needed
        num_unique_imgs = filter_df["image_id"].nunique()
        if num_unique_imgs == 0:
            raise Exception("No image to plot")
        sample_size = min(num_unique_imgs, sample_size)
        sample_imgs_id = filter_df["image_id"].sample(sample_size, random_state=seed).values

        groups = filter_df.loc[filter_df["image_id"].isin(sample_imgs_id)].groupby("image_id")

        self.__plot(
            groups,
            ncols,
            figsize,
            plot_segmentation,
            plot_bbox,
            text_format,
            colors,
            title,
            save_path,
            show,
            seg_kw,
            bbox_kw,
            text_kw,
            **subplot_kw,
        )

    def plot_annotation(
        self,
        anno_id: int | Iterable[int],
        figsize: Iterable[int] | None = None,
        ncols: int = 1,
        title: bool = True,
        plot_segmentation: bool = False,
        plot_bbox: bool = True,
        text_format: str | None = None,
        colors: str | list | dict | None = None,
        save_path: str | None = None,
        show: bool = True,
        seg_kw: dict | None = None,
        bbox_kw: dict | None = None,
        text_kw: dict | None = None,
        **subplot_kw,
    ) -> None:
        """
        Plot annotation(s)

        Parameters
        ----------
        anno_id: int or Iterable[int]
            ID(s) of annotation to plot

        figsize: Iterable[int] of size 2, optional
            Size of matplotlib figure

        ncols : int, default=1
            Number of columns to arrange plot

        title : bool, default=True
            Display image filename as plot title

        plot_segmentation : bool, default=False
            Plot segmentation mask

        plot_bbox : bool, default=True
            Plot bounding box

        text_format : str, optional, default='{anno.name}'
            Format text above bounding box. Access annotation attribute with `anno`

        colors : str, list, dict, optional
            Color of bbox and segmentation mask.
            str and list input will change all to a single color e.g. 'red', [0.5,0.5,0.5]
            dictionary input will set a specific category to a color e.g. {1: "red", 2: [0.5,0.5,0.5]}

        save_path : str, optional
            Path to save figure

        show : bool, default=True
            Plot figure

        seg_kw : dict, optional
            Segmentation annotation plot settings. `alpha`, `linewidth`

        bbox_kw : dict, optional
            Bounding box annotation plot settings. `linewidth`

        text_kw : dict, optional
            Text annotation plot settings. `bbox_alpha`, `text_color`, `fontsize`

        See Also
        --------
        sample_images: sample images to plot
        plot_images: plot specific image(s)
        sample_categories: sample categories to plot
        """
        self.__assert_img_dir()
        self.__assert_attr_not_empty("images")

        if isinstance(anno_id, int):
            anno_id = [anno_id]

        filter_df = self.data[self.data["id"].isin(anno_id)]
        if not len(filter_df):
            raise ValueError("No image to plot")
        sample_imgs_id = filter_df["image_id"].unique()

        groups = filter_df.loc[filter_df["image_id"].isin(sample_imgs_id)].groupby("image_id")

        self.__plot(
            groups,
            ncols,
            figsize,
            plot_segmentation,
            plot_bbox,
            text_format,
            colors,
            title,
            save_path,
            show,
            seg_kw,
            bbox_kw,
            text_kw,
            **subplot_kw,
        )

    # endregion

    # region: write
    @fn_log()
    def to_coco(self, filepath: str, img_dir: str | None = None) -> None:
        """
        Write COCO file

        Parameters
        ----------
        filepath : str
            File path to write

        img_dir : str, optional
            Directory to copy images to. If None, images will not be copied
        """
        if self.annotations.empty:
            logger.error("No data to write")
            return

        output: dict[str, list[dict[str, Any]]] = {
            "categories": [],
            "images": [],
            "annotations": [],
        }

        output["categories"] = json.loads(self.categories.to_json(orient="records"))
        output["images"] = json.loads(self.images.to_json(orient="records"))
        output["annotations"] = json.loads(self.annotations.to_json(orient="records"))

        with open(filepath, "w") as f:
            output_json = json.dumps(output)
            f.write(output_json)

        # copy images
        if img_dir is not None:
            if self.img_dir is None:
                raise ValueError("Copying images requires `img_dir` to be set")

            for _, fpath in self.get_images_mapping().items():
                final_img_dir = os.path.join(img_dir, os.path.dirname(fpath))
                os.makedirs(final_img_dir, exist_ok=True)

            def copy_images(image_fname: str, new_img_dir: str = img_dir) -> None:  # type: ignore
                shutil.copy(os.path.join(self.img_dir, image_fname), os.path.join(new_img_dir, image_fname))  # type: ignore

            image_ids = self.images["file_name"].unique()
            with ProgressBar(total=len(image_ids), desc="Copying images") as pbar, ThreadPoolExecutor(
                max_workers=None
            ) as ex:
                futures = [ex.submit(copy_images, img_fname) for img_fname in image_ids]
                for future in as_completed(futures):
                    pbar.advance()
                pbar.refresh()  # rich progress bar might not complete on it own like tqdm
        return

    @fn_log()
    def to_yolo(
        self,
        directory: str,
        overwrite: bool = False,
        reindex: bool = True,
        copy_images: bool = False,
    ) -> None:
        """
        Write YOLO files

        https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1
        <class> <x_center> <y_center> <width> <height>

        Parameters
        ----------
        directory : str
            Directory to store YOLO files

        overwrite: bool. default=False
            Overwrite directory

        reindex: bool. default=True
            Reindex category IDs to start from 0 and be sequential

        copy_images: bool. default=False
            Copy images to new "images" folder in "directory" argument
        """
        if self.annotations.empty:
            logger.warn("No annotations. There will be no YOLO .txt files generated")

        # check "labels" directory
        label_dir = os.path.join(directory, "labels")
        if os.path.isdir(label_dir) and any(os.scandir(label_dir)):
            if not overwrite:
                raise FileExistsError(f"{label_dir} is not empty")
        else:
            for _, fpath in self.get_images_mapping().items():
                final_label_dir = os.path.join(label_dir, os.path.dirname(fpath))
                os.makedirs(final_label_dir, exist_ok=True)

        # check data.yaml file
        data_yaml_fpath = os.path.join(directory, "data.yaml")
        if os.path.isfile(data_yaml_fpath):
            if not overwrite:
                raise FileExistsError(f"{data_yaml_fpath} already exists")

        # check images directory
        if copy_images:
            if self.img_dir is None:
                raise ValueError("Copying images requires `img_dir` to be set")
            copy_img_dir = os.path.join(directory, "images")
            if os.path.isdir(copy_img_dir) and any(os.scandir(copy_img_dir)):
                if not overwrite:
                    raise FileExistsError(f"{copy_img_dir} is not empty")
            else:
                for _, fpath in self.get_images_mapping().items():
                    final_copy_img_dir = os.path.join(copy_img_dir, os.path.dirname(fpath))
                    os.makedirs(final_copy_img_dir, exist_ok=True)

        # filter necessary info
        cols = ["category_id", "bbox_center_x_p", "bbox_center_y_p", "bbox_w_p", "bbox_h_p", "image_id"]
        if "score" in self.data.columns:
            cols.append("score")
        info = self.data[cols].copy()

        # convert percentages to float
        info[["bbox_center_x_p", "bbox_center_y_p", "bbox_w_p", "bbox_h_p"]] = (
            info[["bbox_center_x_p", "bbox_center_y_p", "bbox_w_p", "bbox_h_p"]] / 100
        )

        # reindex
        if reindex:
            idx_mapping = dict((old_idx, new_idx) for new_idx, old_idx in enumerate(sorted(self.categories["id"])))
            info["category_id"] = info["category_id"].map(idx_mapping)
        else:
            idx_mapping = dict((v, v) for v in sorted(self.categories["id"].values))
        info["category_id"] = info["category_id"].astype(int)

        # write annotations files
        grouped = info.groupby("image_id")
        grouped_img_idx = grouped.groups.keys()

        def write_file(image_id: int, image_copy: bool = copy_images) -> None:
            fname = Path(self.get_images_mapping()[image_id])
            try:
                if image_id in grouped_img_idx:
                    image_df: pd.DataFrame = grouped.get_group(image_id)
                    image_df = image_df.drop("image_id", axis=1)

                    txt_fpath = os.path.join(label_dir, fname.with_suffix(".txt"))
                    image_df.to_csv(
                        txt_fpath,  # type: ignore
                        header=None,  # type: ignore
                        index=None,  # type: ignore
                        sep=" ",
                        mode="w",
                        float_format="%.4g",
                        lineterminator="\n",
                    )

                    # remove last empty line
                    file_data = open(txt_fpath, "r").read()
                    open(txt_fpath, "w").write(file_data.rstrip("\n"))

                # copy images
                if image_copy:
                    shutil.copy(os.path.join(self.img_dir, fname), os.path.join(copy_img_dir, fname))  # type: ignore
            except BaseException:
                raise Exception(f"Error writing {fname.with_suffix('.txt')}")
            return

        image_ids = self.images["id"].unique()
        with ProgressBar(total=len(image_ids), desc="Writing YOLO files") as pbar, ThreadPoolExecutor(
            max_workers=None
        ) as ex:
            futures = [ex.submit(write_file, img_id) for img_id in image_ids]
            for future in as_completed(futures):
                # catch exception from thread
                exception = future.exception()
                if exception:
                    pbar.write(str(exception))
                # progress pbar
                pbar.advance()
            pbar.refresh()  # rich progress bar might not complete on it own like tqdm

        # generate data yaml file
        with open(data_yaml_fpath, "w") as f:
            f.writelines(
                [
                    "path: # root directory\n",
                    "train: # train images (relative to 'path')\n",
                    "val: # val images (relative to 'path')\n",
                    "test: # test images (relative to 'path')\n",
                    f"\nnc: {len(idx_mapping)}\n",
                    "names:\n",
                ]
            )

            for old_id, cat_name in self.get_categories_mapping().items():
                new_id = idx_mapping[old_id]
                f.write(f"  {new_id}: {cat_name}\n")
        return

    @fn_log()
    def to_voc(
        self,
        directory: str,
        overwrite: bool = False,
        prettify: bool = False,
        imagesets_folder: bool = False,
        copy_images: bool = False,
        *,
        folder: str = "",
        database: str = "",
        depth: int = 3,
    ) -> None:
        """
        Write Pascal VOC XML files

        Parameters
        ----------
        directory : str
            Directory to store YOLO files

        overwrite: bool. default=False
            Overwrite directory

        prettify: bool. default=False
            Prettify output with indentation

        imagesets_folder: bool. default=False
            Create ImageSets/Main folder with train.txt, trainval.txt, val.txt and test.txt.
            All images will be written to train (and trainval)

        copy_images: bool. default=False
            Copy images to new "JPEGImages" folder in "directory" argument

        folder: str, default=''
            Value for `folder` element in XML file

        database: str, default=''
            Value for `source.database` element in XML file

        depth: int, default=3
            Value for `size.depth` element in XML file
        """
        if self.images.empty:  # NOTE: empty images requires a XML file, unlike YOLO
            logger.error("No data to write")
            return

        # check "Annotations" directory
        anno_dir = os.path.join(directory, "Annotations")
        if os.path.isdir(anno_dir) and any(os.scandir(anno_dir)):
            if not overwrite:
                raise FileExistsError(f"{anno_dir} is not empty")
        else:
            os.makedirs(anno_dir, exist_ok=True)

        # check images directory
        if copy_images:
            if self.img_dir is None:
                raise ValueError("Copying images requires `img_dir` to be set")
            copy_img_dir = os.path.join(directory, "JPEGImages")
            if os.path.isdir(copy_img_dir) and any(os.scandir(copy_img_dir)):
                if not overwrite:
                    raise FileExistsError(f"{copy_img_dir} is not empty")
            else:
                for _, fpath in self.get_images_mapping().items():
                    final_output_dir = os.path.join(copy_img_dir, os.path.dirname(fpath))
                    os.makedirs(final_output_dir, exist_ok=True)

        # filter necessary info
        cols = ["name", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "image_id", "file_name", "width", "height"]
        if "score" in self.data.columns:
            cols.append("score")
        info = self.data[cols].copy()
        info[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]] = info[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].astype(float)

        # write annotation files
        def write_file(image_id: int) -> None:
            image_df = info[info["image_id"] == image_id]
            image_fname = self.get_images_mapping()[image_id]
            anno_fname = Path(image_fname).with_suffix(".xml")
            fpath = os.path.join(anno_dir, anno_fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)

            # region: XML elements
            root_ele = ET.Element("annotations")

            folder_ele = ET.SubElement(root_ele, "folder")
            folder_ele.text = folder

            filename_ele = ET.SubElement(root_ele, "filename")
            filename_ele.text = image_fname

            path_ele = ET.SubElement(root_ele, "path")
            # NOTE: os.path.join for windows gives \\
            path_ele.text = "/".join([folder, image_fname]) if folder else image_fname

            source_ele = ET.SubElement(root_ele, "source")

            database_ele = ET.SubElement(source_ele, "database")
            database_ele.text = database

            size_ele = ET.SubElement(root_ele, "size")

            width_ele = ET.SubElement(size_ele, "width")
            width_ele.text = str(self.images[self.images["id"] == image_id]["width"].values[0])

            height_ele = ET.SubElement(size_ele, "height")
            height_ele.text = str(self.images[self.images["id"] == image_id]["height"].values[0])

            depth_ele = ET.SubElement(size_ele, "depth")
            depth_ele.text = str(depth)

            segmented_ele = ET.SubElement(root_ele, "segmented")
            # TODO: hardcoded. Assumes image has annotations that are non-linear in shape (polygons)
            segmented_ele.text = "1"

            for anno in image_df.itertuples():
                object_ele = ET.SubElement(root_ele, "object")

                name_ele = ET.SubElement(object_ele, "name")
                name_ele.text = anno.name  # type: ignore[assignment]

                pose_ele = ET.SubElement(object_ele, "pose")
                pose_ele.text = "Unspecified"  # TODO: hardcoded

                truncated_ele = ET.SubElement(object_ele, "truncated")
                truncated_ele.text = "0"  # TODO: hardcoded. Assumes annotation is fully visible

                difficult_ele = ET.SubElement(object_ele, "difficult")
                difficult_ele.text = "0"  # TODO: hardcoded. Assumes annotation is easy

                bndbox_ele = ET.SubElement(object_ele, "bndbox")

                xmin_ele = ET.SubElement(bndbox_ele, "xmin")
                xmin_ele.text = str(round(anno.bbox_x))  # type: ignore

                ymin_ele = ET.SubElement(bndbox_ele, "ymin")
                ymin_ele.text = str(round(anno.bbox_y))  # type: ignore

                xmax_ele = ET.SubElement(bndbox_ele, "xmax")
                xmax_ele.text = str(round(anno.bbox_x + anno.bbox_w))  # type: ignore

                ymax_ele = ET.SubElement(bndbox_ele, "ymax")
                ymax_ele.text = str(round(anno.bbox_y + anno.bbox_h))  # type: ignore

                if "score" in image_df.columns:
                    score_ele = ET.SubElement(bndbox_ele, "score")
                    score_ele.text = str(anno.score)
            # endregion

            # write XML file
            tree = ET.ElementTree(root_ele)
            if prettify:
                ET.indent(tree, space="\t", level=0)
            tree.write(fpath, short_empty_elements=False)

            # copy image
            if copy_images:
                shutil.copy(os.path.join(self.img_dir, image_fname), os.path.join(copy_img_dir, image_fname))  # type: ignore
            return

        image_ids = self.images["id"].unique()
        with ProgressBar(total=len(image_ids), desc="Writing VOC files") as pbar, ThreadPoolExecutor(
            max_workers=None
        ) as ex:
            futures = [ex.submit(write_file, img_id) for img_id in image_ids]
            for future in as_completed(futures):
                # catch exception from thread
                exception = future.exception()
                if exception:
                    pbar.write(str(exception))
                # progress pbar
                pbar.advance()
            pbar.refresh()

        # add ImageSets folder
        if imagesets_folder:
            imagesets_dir = os.path.join(directory, "ImageSets", "Main")
            os.makedirs(imagesets_dir, exist_ok=True)

            # empty txt files for test and val
            with open(os.path.join(imagesets_dir, "test.txt"), "w") as _, open(
                os.path.join(imagesets_dir, "val.txt"), "w"
            ) as _:
                pass

            with open(os.path.join(imagesets_dir, "train.txt"), "w") as t, open(
                os.path.join(imagesets_dir, "trainval.txt"), "w"
            ) as tv:
                image_fnames = self.images["file_name"].unique()
                for image_fname in image_fnames:
                    image_fname = Path(image_fname).stem
                    t.write(image_fname)
                    t.write("\n")
                    tv.write(image_fname)
                    tv.write("\n")
        return

    # endregion

    def __str__(self) -> str:
        imgs_count = len(self.images) if self.images is not None else 0
        cat_count = len(self.categories) if self.categories is not None else 0
        return f"COCOParser: {len(self.annotations):,} annotations. {imgs_count:,} images. {cat_count:,} categories"

    def __repr__(self) -> str:
        return self.__str__()
