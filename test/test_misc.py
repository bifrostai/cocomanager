import pandas as pd
import pytest

import cocomanager as cm


@pytest.fixture
def default_parser():
    return cm.COCOParser("test/test_parse/annotations/coco.json")


def test_cat_mapping(default_parser: cm.COCOParser):
    parser_empty = cm.COCOParser()
    assert parser_empty.get_categories_mapping() == {}

    assert default_parser.get_categories_mapping() == {1: "Car", 2: "Van", 3: "Truck", 4: "Bus"}


def test_img_mapping(default_parser: cm.COCOParser):
    parser_empty = cm.COCOParser()
    assert parser_empty.get_images_mapping() == {}

    assert default_parser.get_images_mapping() == {
        1: "0000000.jpg",
        2: "0000001.jpg",
        3: "0000002.jpg",
        4: "0000003.jpg",
        5: "0000004.jpg",
    }


def test_validate_bbox(default_parser: cm.COCOParser):
    parser_correct = default_parser
    parser_wrong = cm.COCOParser("test/test_misc/fixtures/error_bbox.json")

    assert parser_correct.validate_bbox(show_result=False) is True
    assert parser_wrong.validate_bbox(show_result=False) is False

    error_df = pd.DataFrame(
        [
            [1, "0000000.jpg", 100, 100, -10, 20, 60, 30],
            [2, "0000000.jpg", 100, 100, 20, 50, 100, 20],
            [3, "0000001.jpg", 200, 200, 10, -20, 60, 40],
            [4, "0000001.jpg", 200, 200, 40, 20, 60, 200],
            [5, "0000001.jpg", 200, 200, -10, -20, 500, 500],
        ],
        columns=["id", "file_name", "width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
    )
    assert parser_wrong.validate_bbox(show_result=True).equals(error_df)

    style_txt = open("test/test_misc/results/error_bbox_style.html", "r").read()
    assert parser_wrong.validate_bbox(show_result=True, style=True).to_html(uuid="fix_id") == style_txt


def test_copy(default_parser: cm.COCOParser):
    parser_copy = default_parser.copy()

    assert parser_copy.annotations.equals(default_parser.annotations)
    assert parser_copy.categories.equals(default_parser.categories)
    assert parser_copy.images.equals(default_parser.images)
    assert parser_copy.img_dir == default_parser.img_dir
