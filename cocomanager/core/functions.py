import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import pandas as pd
import ujson as json
from PIL import Image
from pydantic import ValidationError

from cocomanager.core.parser import COCOParser
from cocomanager.models.coco import (
    CocoCategory,
    CocoDetectionAnnotation,
    CocoImages,
    CocoInfo,
    CocoLicense,
)

from ..utils.logger import fn_log
from ..utils.progressbar import ProgressBar


@fn_log()
def concat(
    data: Sequence[COCOParser],
    *,
    images_handling: str = "suffix",
    categories_handling: str = "strict",
    fname_extension: Sequence[str] | None = None,
    new_img_dir: str | None = None,
) -> COCOParser:
    """
    Concatenate COCOParsers
    Allows setting of logic to avoid duplicate image file names and inconsistent categories ID to name mapping

    Parameters
    ----------
    data : Iterable[COCOParser]
        Sequence of COCOParser instances to concatentate

    images_handling : {"suffix", "prefix", "increment", "retain"}
        Logic to handle image file names to prevent duplicate file name with different image ID.
        default=suffix
        `suffix`: add suffix at the end of image file name, based on `fname_extension`
        `prefix`: add prefix at the start of image file name, based on `fname_extension`
        `increment`: only works if file name can be parsed as an integer. adds 1 to image file name and lpad with 0 to match initial length
        `retain`: file_name remains unchanged. image ID will be set to the same value if file_name is the same

    categories_handling : {"strict", "lenient", "extend"}
        Logic to handle categories name and ID.
        `strict`: requires all category ID to name mapping to be exactly the same
        `lenient`: requires the number of categories and categories names to be the same. ID to name mapping will be adjusted when needed
        `extend`: each category name will be assigned its own category ID

    fname_extension: Iterable[str], optional
        Suffix or prefix added to file_name for each COCOParser. Length of `fname_extension` must be the same as `data`

    new_img_dir: str, optional
        Directory to copy the images with updated file names

    Returns
    -------
    COCOParser
    """
    # check images_handling & fname_extension arguments
    valid_img_handling = ["suffix", "prefix", "increment", "retain"]
    if images_handling not in valid_img_handling:
        raise ValueError(f"{images_handling} is not a valid option. Select from {valid_img_handling}")
    if images_handling in ["suffix", "prefix"]:
        if fname_extension is None:
            raise ValueError(f"{images_handling} handling for images requires `fname_extension`")
        if len(fname_extension) != len(data):
            raise ValueError(
                f"fname_extension length ({len(fname_extension)}) is not equals to data length ({len(data)})"
            )

    # check categories_handling argument
    valid_category_handling = ["extend", "strict", "lenient"]
    if categories_handling not in valid_category_handling:
        raise ValueError(f"{categories_handling} is not a valid option. Select from {valid_category_handling}")

    # check new_img_dir argument
    if new_img_dir is not None:
        for coco_parser in data:
            if coco_parser.img_dir is None:
                raise Exception("Copying images to new_img_dir requires all COCOParser to have an img_dir attribute")

    overall_images = pd.DataFrame()
    overall_cat = pd.DataFrame()
    overall_anno = pd.DataFrame()
    cat_name2id: dict[str, int] = {}
    img_name2id: dict[str, int] = {}

    with ProgressBar(total=len(data), desc="Concatenating datasets") as pbar:
        for idx, coco_parser in enumerate(data):
            tmp_images = coco_parser.images.copy()
            tmp_cat = coco_parser.categories.copy()
            tmp_anno = coco_parser.annotations.copy()

            # category
            tmp_cat_id2name = coco_parser.get_categories_mapping()
            tmp_cat_name2id = dict((v, k) for k, v in tmp_cat_id2name.items())

            if not cat_name2id:
                cat_name2id = tmp_cat_name2id
            elif categories_handling == "strict":
                if cat_name2id != tmp_cat_name2id:
                    cat_id2name = dict((v, k) for k, v in cat_name2id.items())
                    raise Exception(f"Inconsistent categories mapping found. {cat_id2name} vs {tmp_cat_id2name}")
            elif categories_handling == "lenient":
                cat_old = set(cat_name2id.keys())
                cat_new = set(tmp_cat_name2id.keys())
                if cat_old != cat_new:
                    cat_diff = list(cat_old.symmetric_difference(cat_new))
                    raise Exception(f"Different categories found {cat_diff}")
            else:
                for cat_name in tmp_cat_name2id.keys():
                    if cat_name not in cat_name2id:
                        cat_name2id[cat_name] = max(cat_name2id.values()) + 1

            tmp_cat["id"] = tmp_cat["name"].map(cat_name2id)

            new_id2id = dict((k, cat_name2id[v]) for k, v in tmp_cat_id2name.items())
            tmp_anno["category_id"] = tmp_anno["category_id"].map(new_id2id)

            # image
            def update_fname(fname: str, mode: str, change: str | int, id_count: int) -> str:
                """helper function to convert image filename accordingly"""
                p = Path(fname)
                parent_dir = "" if str(p.parent) == "." else f"{p.parent}/"

                if mode == "suffix":
                    return f"{parent_dir}{p.stem}{change}{p.suffix}"
                elif mode == "prefix":
                    return f"{parent_dir}{change}{p.stem}{p.suffix}"
                elif mode == "increment":
                    LENGTH = 7
                    new_fname = str(change + id_count)  # type: ignore
                    return f"{parent_dir}{new_fname.zfill(LENGTH)}{p.suffix}"
                else:
                    return fname

            def copy_file(source_dir: str, source_fname: str, dest_dir: str, dest_fname: str) -> None:
                """Helper function to move file"""
                source = os.path.join(source_dir, source_fname)
                dest = os.path.join(dest_dir, dest_fname)
                os.makedirs(os.path.dirname(dest), exist_ok=True)  # do it here as dest_fname might contain a directory
                shutil.copy(source, dest)

            # get mapping of old file name to new file name
            tmp_img_id2name = coco_parser.get_images_mapping()
            if images_handling in ["suffix", "prefix"]:
                change = fname_extension[idx]  # type: ignore
            else:
                change = len(img_name2id)
            update_img_fname_map = dict(
                (fname, update_fname(fname, images_handling, change, id_count))
                for id_count, fname in enumerate(tmp_img_id2name.values())
            )

            # update img_name2id
            if not img_name2id:
                img_name2id = dict((v, i) for i, v in enumerate(update_img_fname_map.values(), 1))
            elif images_handling == "retain":
                # TODO: check image width and height
                for img_fname in update_img_fname_map.values():
                    if img_fname not in img_name2id:
                        img_name2id[img_fname] = max(img_name2id.values()) + 1
            else:
                # ensure image file name does not repeat
                repeated_fnames = set(img_name2id.keys()).intersection(set(update_img_fname_map.values()))
                if repeated_fnames:
                    shortened_list = list(repeated_fnames)[:5]
                    if len(repeated_fnames) > 5:
                        shortened_list.append("...")
                    raise Exception(f"Repeat file name found: {shortened_list}")
                else:
                    for img_fname in update_img_fname_map.values():
                        img_name2id[img_fname] = len(img_name2id) + 1

            tmp_images["file_name"] = tmp_images["file_name"].map(update_img_fname_map)
            tmp_images["id"] = tmp_images["file_name"].map(img_name2id)

            new_id2id = dict((k, img_name2id[update_img_fname_map[v]]) for k, v in tmp_img_id2name.items())
            tmp_anno["image_id"] = tmp_anno["image_id"].map(new_id2id)

            # copy files
            if new_img_dir is not None:
                image_files = tmp_img_id2name.values()
                if not pbar.in_notebook:
                    img_task = pbar.add_task("Copying images", visible=False, total=len(image_files))
                else:
                    # TODO: add secondary tqdm bar showing progress of copying images
                    # img_tqdm_bar = ProgressBar(total=len(image_files), desc="Copying images", keep_alive=False)
                    pass
                with ThreadPoolExecutor(max_workers=1) as ex:
                    futures = [
                        ex.submit(copy_file, coco_parser.img_dir, fname, new_img_dir, update_img_fname_map[fname])  # type: ignore
                        for fname in image_files
                    ]
                    for future in as_completed(futures):
                        if not pbar.in_notebook:
                            pbar.update(task=img_task, visible=sum(f.done() for f in futures) < len(image_files))
                        else:
                            # TODO: add secondary tqdm bar showing progress of copying images
                            # img_tqdm_bar.update(sum(f.done() for f in futures) - img_tqdm_bar.tqdm_bar.n)
                            pass

            # concat dataframe
            overall_images = pd.concat([overall_images, tmp_images], ignore_index=True)
            overall_cat = pd.concat([overall_cat, tmp_cat], ignore_index=True)
            overall_anno = pd.concat([overall_anno, tmp_anno], ignore_index=True)

            pbar.advance()
        pbar.refresh()

    # reset annotation ID
    overall_anno["id"] = range(1, len(overall_anno) + 1)

    # make a new COCOParser
    output_parser = COCOParser()

    # add attributes
    output_parser.images = overall_images.drop_duplicates(["id", "file_name"])
    output_parser.categories = overall_cat.drop_duplicates(["id", "name"])
    output_parser.annotations = overall_anno
    output_parser.img_dir = new_img_dir

    return output_parser


@fn_log()
def annotate_all_images(
    parser: COCOParser,
    output_dir: str,
    plot_bbox: bool = True,
    plot_seg: bool = False,
    workers: int | None = 1,
) -> None:
    """
    Plot annotations on all images in the dataset

    Parameters
    ----------
    parser : COCOParser
        COCOParser object

    output_dir : str
        Output directory to save the annotated images

    plot_bbox : bool, optional, default = True
        Plot bounding boxes

    plot_seg : bool, optional, default = False
        Plot segmentation

    workers : int, optional, default = 1
        Number of workers to use for multi-threading

    Returns
    -------
    None
    """
    if parser.img_dir is None:
        raise AttributeError("Provide img_dir attribute to annotate images")

    matplotlib.use("Agg")

    img_fnames = parser.images["file_name"].values
    img_widths = parser.images["width"].values
    img_heights = parser.images["height"].values

    for _, fpath in parser.get_images_mapping().items():
        final_output_dir = os.path.join(output_dir, os.path.dirname(fpath))
        os.makedirs(final_output_dir, exist_ok=True)

    with ProgressBar(total=len(img_fnames), desc="Annotating images") as pbar, ThreadPool(workers) as pool:
        try:
            for img_fname, img_width, img_height in zip(img_fnames, img_widths, img_heights):
                img_fpath = os.path.join(parser.img_dir, img_fname)
                if not os.path.exists(img_fpath):
                    pbar.write(f"Image file not found: {img_fpath}")
                    pbar.advance()
                    continue
                dpi: int = Image.open(img_fpath).info.get("dpi", (96, 96))[0]
                text_size = 10 * (96 / dpi)
                bbox_lw = 1.5 * (96 / dpi)
                seg_lw = 2 * (96 / dpi)
                pool.apply_async(
                    parser.plot_images,
                    args=(
                        img_fname,
                        (img_width / dpi, img_height / dpi),
                        1,  # ncols
                        False,  # title
                        plot_seg,
                        plot_bbox,
                    ),
                    kwds={
                        "dpi": dpi,
                        "save_path": os.path.join(output_dir, img_fname),
                        "show": False,
                        "frameon": "off",
                        "text_kw": {"fontsize": text_size},
                        "bbox_kw": {"linewidth": bbox_lw},
                        "seg_kw": {"linewidth": seg_lw},
                    },
                    callback=lambda _: pbar.advance(),
                    error_callback=print,
                ).get(200)
        except KeyboardInterrupt:
            pbar.write("KeyboardInterrupt")
            pool.terminate()
            pool.join()
        pool.close()
        pool.join()
    return


def validatate_coco(__data: str | dict, /, *, verbose: bool = False, log_file: str | None = None) -> bool:
    """
    Validate `images`, `categories` and `annotations` of a COCO json file

    Parameters
    ----------
    __data: str or dictionary
        File path or dictionary representation of the COCO annotations

    verbose: bool, default=False
        To print validations errors on console or not

    log_file: str, default=None
        Write output to file

    Returns
    -------
    Boolean stating if the COCO annotations are valid or not
    """
    valid = True
    if isinstance(__data, str):
        coco_data: dict[str, Any] = json.load(open(__data, "r"))
    elif isinstance(__data, dict):
        coco_data = __data
    else:
        raise TypeError(f"Invalid input type: {type(__data)}")

    def _output(msg: str = "") -> None:
        if verbose:
            print(msg)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(msg)
                f.write("\n")

    def _format_val_error(e: ValidationError, idx: int, cls_name: str) -> str:
        """
        Format pydantic ValidationError
        """
        error_msg = str(e)
        # format is ["n validation errors for COCO..", "<attr>", "<attr error>", ..]
        error_msg_list = error_msg.split("\n")

        output_msg_list: list[str] = []
        # replace "for COCO.."" with "at index <idx>"
        output_msg_list.append(error_msg_list[0].replace(f"for {cls_name}", f"at index {idx}"))

        # concat attribute and its error message
        for attr, er in zip(*[iter(error_msg_list[1:])] * 2):  # type: ignore
            pattern = r"\(type=.*\)$"
            er = re.sub(pattern, "", er)
            output_msg_list.append(f"{attr}: {er.strip()}")

        output_msg = "\n    ".join(output_msg_list)

        return f"  {output_msg}"

    def _validate_info(infos: list[dict[str, Any]]) -> bool:
        """
        Validate info
            - Invalid info (error)
        """
        _output("Validating info..")
        valid = True

        # check each annotation entry
        for idx, info in enumerate(infos):
            try:
                CocoInfo(**info)
            except ValidationError as ve:
                valid = False
                msg = _format_val_error(ve, idx, "COCOInfo")
                _output(msg)
                _output()
            except Exception as e:
                _output(str(e))

        return valid

    def _validate_categories(categories: list[dict[str, Any]]) -> bool:
        """
        Validate categories
            - Invalid category format (error)
            - Repeated IDs (error)
            - Starting index is not 1 (warning)
            - Index not sequential (warning)
            - Repeated names (warning)
        """
        _output("Validating categories..")
        valid = True
        seen_ids: set[int] = set()
        repeated_ids: set[int] = set()

        name2id: dict[str, set[int]] = {}

        # check each category entry
        for idx, category in enumerate(categories):
            try:
                cat = CocoCategory(**category)
                if cat.id in seen_ids:
                    repeated_ids.add(cat.id)
                else:
                    seen_ids.add(cat.id)
                if cat.name in name2id:
                    name2id[cat.name].add(cat.id)
                else:
                    name2id[cat.name] = set([cat.id])
            except ValidationError as ve:
                valid = False
                msg = _format_val_error(ve, idx, "COCOCategories")
                _output(msg)
                _output()
            except Exception as e:
                _output(str(e))

        # error if repeated IDs
        if repeated_ids:
            valid = False
            _output(f"  Category ID(s) {list(repeated_ids)} are repeated\n")

        # warn if category IDs are not sequential
        if seen_ids:
            starting_index = min(seen_ids)
            if starting_index != 1:
                _output(f"  WARNING: Category index starts at {starting_index}")
            if seen_ids != set(range(starting_index, starting_index + len(seen_ids))):
                _output("  WARNING: Category indexes are not sequential")

        # warn if category name appears for different ID
        for name, ids in name2id.items():
            if len(ids) > 1:
                _output(f"  WARNING: Category name {name} is repeated for multiple IDs: {list(ids)}")

        return valid

    def _validate_images(images: list[dict[str, Any]], licenses: set[int] | None) -> bool:
        """
        Validate images
            - Invalid image format (error)
            - Repeated IDs (error)
            - Repeated file names (error)
            - Starting index is not 1 (warning)
            - Index not sequential (warning)
        """
        _output("Validating images..")
        valid = True
        seen_ids: set[int] = set()
        repeated_ids: set[int] = set()
        seen_names: set[str] = set()
        repeated_names: set[str] = set()

        # check each image entry
        for idx, image_data in enumerate(images):
            try:
                image = CocoImages(**image_data)
                if image.id in seen_ids:
                    repeated_ids.add(image.id)
                else:
                    seen_ids.add(image.id)
                if image.file_name in seen_names:
                    repeated_names.add(image.file_name)
                else:
                    seen_names.add(image.file_name)

                if image.license is not None and (licenses is None or image.license not in licenses):
                    _output(f"  1 validataion error at index {idx}")
                    _output(f"    license: unable to find license ID {image.license}")
            except ValidationError as ve:
                valid = False
                msg = _format_val_error(ve, idx, "COCOImages")
                _output(msg)
                _output()
            except Exception as e:
                _output(str(e))

        # error if repeated IDs
        if repeated_ids:
            valid = False
            _output(f"  Image ID(s) {list(repeated_ids)} are repeated\n")

        # error if repeated file name
        if repeated_names:
            valid = False
            _output(f"  Image file name(s) {list(repeated_names)} are repeated\n")

        # warn if IDs are not sequential
        if seen_ids:
            starting_index = min(seen_ids)
            if starting_index != 1:
                _output(f"  WARNING: Image index starts at {starting_index}")
            if seen_ids != set(range(starting_index, starting_index + len(seen_ids))):
                _output("  WARNING: Image indexes are not sequential")

        return valid

    def _validate_licenses(licenses: list[dict[str, Any]]) -> tuple[bool, set[int] | None]:
        """
        Validate licenses
            - Invalid license format (error)
            - Repeated IDs (error)
        """
        _output("Validating licenses..")
        valid = True
        seen_ids: set[int] = set()
        repeated_ids: set[int] = set()

        # check each image entry
        for idx, license_data in enumerate(licenses):
            try:
                license_ = CocoLicense(**license_data)
                if license_.id in seen_ids:
                    repeated_ids.add(license_.id)
                else:
                    seen_ids.add(license_.id)
            except ValidationError as ve:
                valid = False
                msg = _format_val_error(ve, idx, "COCOLicense")
                _output(msg)
                _output()
            except Exception as e:
                _output(str(e))

        # error if repeated IDs
        if repeated_ids:
            valid = False
            _output(f"  License ID(s) {list(repeated_ids)} are repeated\n")

        # warn if IDs are not sequential
        if seen_ids:
            starting_index = min(seen_ids)
            if starting_index != 1:
                _output(f"  WARNING: Image index starts at {starting_index}")
            if seen_ids != set(range(starting_index, starting_index + len(seen_ids))):
                _output("  WARNING: Image indexes are not sequential")

        return (valid, seen_ids if valid else None)

    def _validate_annotations(annotations: list[dict[str, Any]]) -> bool:
        """
        Validate annotations
            - Invalid annotation (error)
            - Repeated IDs (error)
            - Invalid category (error) TODO
            - Invalid image (error) TODO
        """
        _output("Validating annotations..")
        valid = True
        seen_ids: set[int] = set()
        repeated_ids: set[int] = set()

        # check each annotation entry
        for idx, anno_data in enumerate(annotations):
            try:
                anno = CocoDetectionAnnotation(**anno_data)
                if anno.id in seen_ids:
                    repeated_ids.add(anno.id)
                else:
                    seen_ids.add(anno.id)
            except ValidationError as ve:
                valid = False
                msg = _format_val_error(ve, idx, "COCOAnnotation")
                _output(msg)
                _output()
            except Exception as e:
                _output(str(e))

        # error if repeated IDs
        if repeated_ids:
            valid = False
            _output(f"  Annotation ID(s) {list(repeated_ids)} are repeated\n")

        # warn if IDs are not sequential
        if seen_ids:
            starting_index = min(seen_ids)
            if starting_index != 1:
                _output(f"  WARNING: Annotation index starts at {starting_index}")
            if seen_ids != set(range(starting_index, starting_index + len(seen_ids))):
                _output("  WARNING: Annotation indexes are not sequential")

        return valid

    # info
    if coco_data.get("info") is None:
        _output("Skipping validation for info")
    else:
        info_valid = _validate_info(coco_data["info"])
        valid = info_valid if not info_valid else valid
    _output()

    # licenses
    if coco_data.get("licenses") is None:
        _output("Skipping validation for licenses")
        license_ids = None
    else:
        license_valid, license_ids = _validate_licenses(coco_data["licenses"])
        valid = license_valid if not license_valid else valid
    _output()

    # images
    if coco_data.get("images") is None:
        valid = False
        _output("Missing essential key: images")
    else:
        img_valid = _validate_images(coco_data["images"], license_ids)
        valid = img_valid if not img_valid else valid
    _output()

    # categories
    if coco_data.get("categories") is None:
        # NOTE: not required
        _output("Skipping validation for categories")
    else:
        cat_valid = _validate_categories(coco_data["categories"])
        valid = cat_valid if not cat_valid else valid
    _output()

    # annotations
    if coco_data.get("annotations") is None:
        valid = False
        _output("Missing essential key: annotations")
    else:
        anno_valid = _validate_annotations(coco_data["annotations"])
        valid = anno_valid if not anno_valid else valid
    _output()

    return valid
