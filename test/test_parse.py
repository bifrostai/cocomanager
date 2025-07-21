from ast import literal_eval

import pandas as pd
import pytest

import cocomanager as cm


def load_test_data(format: str):
    anno_df = pd.read_csv(
        f"test/test_parse/results/{format}_annotations.csv",
        converters={"bbox": literal_eval, "segmentation": literal_eval, "area": literal_eval},
    )
    image_df = pd.read_csv(f"test/test_parse/results/{format}_images.csv")
    cat_df = pd.read_csv(f"test/test_parse/results/{format}_categories.csv")
    cat_df["name"] = cat_df["name"].astype(str)  # pandas will convert the name column to int type

    return anno_df, image_df, cat_df


def test_parse_empty():
    parser = cm.COCOParser()

    assert parser.annotations.empty
    assert parser.images.empty
    assert parser.categories.empty


def test_parse_missing_file():
    with pytest.raises(FileNotFoundError):
        cm.COCOParser("test/test_parse/missing/file")


def test_parse_coco():
    # correct dataframes
    anno_df, image_df, cat_df = load_test_data("coco")

    # parse COCO annotations
    parser = cm.COCOParser("test/test_parse/annotations/coco.json")

    # compare annotations
    assert anno_df.equals(parser.annotations)
    assert image_df.equals(parser.images)
    assert cat_df.equals(parser.categories)


def test_parse_voc():
    # correct dataframes
    anno_df, image_df, cat_df = load_test_data("voc")

    # parse VOC annotations
    parser = cm.COCOParser.parse_from_voc("test/test_parse/annotations/voc")

    # compare annotations
    assert anno_df.equals(parser.annotations)
    assert image_df.equals(parser.images)
    assert cat_df.equals(parser.categories)


def test_parse_yolo():
    # correct dataframes
    anno_df, image_df, cat_df = load_test_data("yolo")
    _, _, coco_cat_df = load_test_data("coco")
    coco_cat_df["id"] -= 1  # COCO starts at 1, YOLO starts at 0

    # parse VOC annotations
    without_cat_map = cm.COCOParser.parse_from_yolo("test/test_parse/annotations/yolo/labels", "test/test_parse/images")
    with_cat_map = cm.COCOParser.parse_from_yolo(
        "test/test_parse/annotations/yolo/labels",
        "test/test_parse/images",
        category_map="test/test_parse/annotations/yolo/data.yaml",
    )

    # compare annotations
    assert anno_df.equals(without_cat_map.annotations)
    assert image_df.equals(without_cat_map.images)
    assert cat_df.equals(without_cat_map.categories)

    assert coco_cat_df.equals(with_cat_map.categories)
