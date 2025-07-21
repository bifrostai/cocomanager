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
def parser() -> cm.COCOParser:
    return cm.COCOParser("test/test_parse/annotations/coco.json", "test/test_parse/images")


@pytest.mark.parametrize("img_dir", [None, "test/tmp/images"])
def test_to_coco(parser: cm.COCOParser, tmp_path: Literal["test/tmp"], img_dir: str | None):
    parser.to_coco(f"{tmp_path}/coco.json", img_dir=img_dir)

    # check file counts
    if img_dir is None:
        assert len(glob(f"{tmp_path}/*")) == 1
    else:
        assert len(glob(f"{tmp_path}/*")) == 2
        assert len(glob(f"{tmp_path}/images/*")) == 5

    # compare source json
    cmp_coco_json = filecmp.cmpfiles(
        tmp_path,
        "test/test_parse/annotations",
        ["coco.json"],
        shallow=False,
    )
    assert len(cmp_coco_json[1]) == 0, f"Files mismatched: {cmp_coco_json[1]}"
    assert len(cmp_coco_json[2]) == 0, f"Files errors: {cmp_coco_json[2]}"

    # compare images
    if img_dir is not None:
        cmp_imgs = filecmp.cmpfiles(
            f"{tmp_path}/images",
            "test/test_parse/images",
            ["0000000.jpg", "0000001.jpg", "0000002.jpg", "0000003.jpg", "0000004.jpg"],
            shallow=False,
        )
        assert len(cmp_imgs[1]) == 0, f"Files mismatched: {cmp_imgs[1]}"
        assert len(cmp_imgs[2]) == 0, f"Files errors: {cmp_imgs[2]}"


def test_to_voc(parser: cm.COCOParser, tmp_path: Literal["test/tmp"]):
    parser.to_voc(tmp_path, imagesets_folder=True, copy_images=True, overwrite=False)

    # check directories generated
    assert len(glob(f"{tmp_path}/*")) == 3

    # Annotations
    assert len(glob(f"{tmp_path}/Annotations/*")) == 5
    cmp_imgs = filecmp.cmpfiles(
        f"{tmp_path}/Annotations",
        "test/test_write/results/voc/Annotations",
        ["0000000.xml", "0000001.xml", "0000002.xml", "0000003.xml", "0000004.xml"],
        shallow=False,
    )
    assert len(cmp_imgs[1]) == 0, f"Files mismatched: {cmp_imgs[1]}"
    assert len(cmp_imgs[2]) == 0, f"Files errors: {cmp_imgs[2]}"

    # ImageSets
    assert len(glob(f"{tmp_path}/ImageSets/*")) == 1
    assert len(glob(f"{tmp_path}/ImageSets/Main/*")) == 4
    cmp_imgs = filecmp.cmpfiles(
        f"{tmp_path}/ImageSets/Main",
        "test/test_write/results/voc/ImageSets/Main",
        ["test.txt", "train.txt", "trainval.txt", "val.txt"],
        shallow=False,
    )
    assert len(cmp_imgs[1]) == 0, f"Files mismatched: {cmp_imgs[1]}"
    assert len(cmp_imgs[2]) == 0, f"Files errors: {cmp_imgs[2]}"

    # Images
    assert len(glob(f"{tmp_path}/JPEGImages/*")) == 5
    cmp_imgs = filecmp.cmpfiles(
        f"{tmp_path}/JPEGImages",
        "test/test_parse/images",
        ["0000000.jpg", "0000001.jpg", "0000002.jpg", "0000003.jpg", "0000004.jpg"],
        shallow=False,
    )
    assert len(cmp_imgs[1]) == 0, f"Files mismatched: {cmp_imgs[1]}"
    assert len(cmp_imgs[2]) == 0, f"Files errors: {cmp_imgs[2]}"

    # test without overwriting
    with pytest.raises(FileExistsError, match=f"^{tmp_path}\\\\Annotations is not empty$"):
        parser.to_voc(tmp_path, imagesets_folder=True, copy_images=True, overwrite=False)


def test_to_yolo(parser: cm.COCOParser, tmp_path: Literal["test/tmp"]):
    parser.to_yolo(tmp_path, copy_images=True, overwrite=False, reindex=False)

    # check directories generated
    assert len(glob(f"{tmp_path}/*")) == 3

    # data.yaml
    cmp_data = filecmp.cmpfiles(
        f"{tmp_path}",
        "test/test_write/results/yolo",
        ["data.yaml"],
        shallow=False,
    )
    assert len(cmp_data[1]) == 0, f"Files mismatched: {cmp_data[1]}"
    assert len(cmp_data[2]) == 0, f"Files errors: {cmp_data[2]}"

    # Annotations
    assert len(glob(f"{tmp_path}/labels/*")) == 5
    cmp_labels = filecmp.cmpfiles(
        f"{tmp_path}/labels",
        "test/test_write/results/yolo/labels",
        ["0000000.txt", "0000001.txt", "0000002.txt", "0000003.txt", "0000004.txt"],
        shallow=False,
    )
    assert len(cmp_labels[1]) == 0, f"Files mismatched: {cmp_labels[1]}"
    assert len(cmp_labels[2]) == 0, f"Files errors: {cmp_labels[2]}"

    # Images
    assert len(glob(f"{tmp_path}/images/*")) == 5
    cmp_imgs = filecmp.cmpfiles(
        f"{tmp_path}/images",
        "test/test_parse/images",
        ["0000000.jpg", "0000001.jpg", "0000002.jpg", "0000003.jpg", "0000004.jpg"],
        shallow=False,
    )
    assert len(cmp_imgs[1]) == 0, f"Files mismatched: {cmp_imgs[1]}"
    assert len(cmp_imgs[2]) == 0, f"Files errors: {cmp_imgs[2]}"

    # test without overwriting
    with pytest.raises(FileExistsError, match=f"^{tmp_path}\\\\labels is not empty$"):
        parser.to_yolo(tmp_path, copy_images=True, overwrite=False)
