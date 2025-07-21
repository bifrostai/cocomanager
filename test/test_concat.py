# TODO: improve test coverage
import filecmp
import os
import shutil
from glob import glob
from typing import Literal

import pytest

import cocomanager as cm


def clear_tmp():
    """Clear tmp folder"""
    for path in glob("test/tmp/*"):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


@pytest.fixture(scope="function", autouse=True)
def setup_function():
    """Clear tmp folder before each test"""
    clear_tmp()


@pytest.fixture(scope="session", autouse=True)
def teardown_session():
    """Clear tmp folder after all tests"""
    yield
    clear_tmp()


@pytest.fixture
def tmp_path() -> Literal["test/tmp"]:
    return "test/tmp"


@pytest.fixture
def base_parser() -> cm.COCOParser:
    return cm.COCOParser("test/test_parse/annotations/coco.json", "test/test_parse/images")


@pytest.mark.parametrize(
    "anno_file",
    [
        "test/test_concat/fixtures/annotations/coco_diff_cat_order.json",
        "test/test_concat/fixtures/annotations/coco_diff_cat_name.json",
    ],
)
def test_concat(base_parser: cm.COCOParser, tmp_path: Literal["test/tmp"], anno_file: str):
    additional_parser = cm.COCOParser(anno_file, "test/test_concat/fixtures/images")

    new_img_dir = f"{tmp_path}/images"
    concat_result = cm.concat(
        [additional_parser, base_parser],
        images_handling="suffix",
        categories_handling="extend",
        fname_extension=["_a", "_b"],
        new_img_dir=new_img_dir,
    )

    # check COCOParser result
    result_path = f"test/test_concat/results/annotations/{os.path.basename(anno_file)}"
    result_parser = cm.COCOParser(result_path)
    assert concat_result.img_dir == new_img_dir
    assert concat_result.annotations.equals(result_parser.annotations)
    assert concat_result.categories.equals(result_parser.categories)
    assert concat_result.images.equals(result_parser.images)

    # check images
    assert len(glob(f"{tmp_path}/*")) == 1
    assert len(glob(f"{new_img_dir}/*")) == 10

    img_result_dir = "test/test_concat/results/images"
    cmp_imgs = filecmp.cmpfiles(
        img_result_dir,
        new_img_dir,
        os.listdir(img_result_dir),
        shallow=False,
    )
    assert len(cmp_imgs[1]) == 0, f"Files mismatched: {cmp_imgs[1]}"
    assert len(cmp_imgs[2]) == 0, f"Files errors: {cmp_imgs[2]}"
