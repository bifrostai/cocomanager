import json

import pytest
from pydantic import ValidationError

import cocomanager as cm


def test_clip_bbox():
    parser = cm.COCOParser("test/test_misc/fixtures/error_bbox.json")
    output = parser.clip_bbox()
    assert output.annotations["bbox"].values.tolist() == [
        [0, 20, 50, 30],
        [20, 50, 80, 20],
        [10, 0, 60, 20],
        [40, 20, 60, 180],
        [0, 0, 200, 200],
        [10, 20, 50, 40],
    ]


@pytest.mark.parametrize("use_parser", [True, False])
@pytest.mark.parametrize("include_extra", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("show_changes", [True, False])
def test_match_categories(
    use_parser: bool, include_extra: bool, inplace: bool, show_changes: bool, capsys: pytest.CaptureFixture[str]
):
    parser = cm.COCOParser("test/test_parse/annotations/coco.json")
    other = (
        cm.COCOParser("test/test_edit/fixtures/match_categories_coco.json")
        if use_parser
        else {1: "Lorry", 2: "Bus", 3: "Van", 4: "Truck", 5: "Car"}
    )

    names = ["Car", "Van", "Truck", "Bus"]
    old_ids: list[int | None] = [1, 2, 3, 4]
    new_ids = [5, 3, 4, 2]
    if include_extra:
        names = [*names, "Lorry"]
        old_ids = [*old_ids, None]
        new_ids = [*new_ids, 1]

    output = parser.match_categories(other, include_extra=include_extra, inplace=inplace, show_changes=show_changes)

    captured = capsys.readouterr()
    if show_changes:
        expected_changes = {}
        for name, old_id, new_id in zip(names, old_ids, new_ids):
            if old_id != new_id:
                expected_changes[name] = {"old": old_id, "new": new_id}

        expected_output = str(expected_changes)
        assert captured.out.strip() == expected_output
    else:
        assert captured.out.strip() == ""

    assert set(output.categories["name"].values.tolist()) == set(names)
    assert set(output.categories["id"].values.tolist()) == set(new_ids)


def test_rename_categories():
    # {1: 'Car', 2: 'Van', 3: 'Truck', 4: 'Bus'}
    parser = cm.COCOParser("test/test_parse/annotations/coco.json")
    output = parser.rename_categories({"Van": "Car", "Bus": "Baz", "Foo": "Bar"})
    assert output.get_categories_mapping() == {1: "Car", 3: "Truck", 4: "Baz"}


@pytest.mark.parametrize("update_type", ["correct", "incorect_cat", "incorect_bbox"])
def test_update_annotations(update_type: str):
    parser = cm.COCOParser("test/test_edit/fixtures/update_annotations_coco.json")

    update: dict = json.load(open("test/test_edit/fixtures/updates.json"))[update_type]
    update = {int(k): v for k, v in update.items()}

    if update_type == "correct":
        output = parser.update_annotations(update)

        result = cm.COCOParser("test/test_edit/results/update_annotations_coco.json")
        assert output.annotations.equals(result.annotations)

    elif update_type == "incorect_cat":
        with pytest.raises(ValidationError, match=r"Category 5 is not within category list: \[1, 2, 3\]"):
            output = parser.update_annotations(update)

    else:
        with pytest.raises(
            ValidationError,
            match=r"List should have at least 4 items",
        ):
            output = parser.update_annotations(update)


@pytest.mark.parametrize("action", ["remove", "filter"])
@pytest.mark.parametrize("drop_empty_images", [True, False])
def test_remove_filter_anno(action: str, drop_empty_images: bool):
    parser = cm.COCOParser("test/test_edit/fixtures/remove_filter_coco.json")

    # perform remove/filter
    anno_ids = [1, 2, 3]
    output = (
        parser.remove_annotations(anno_ids, drop_empty_images=drop_empty_images)
        if action == "remove"
        else parser.filter_annotations(anno_ids, drop_empty_images=drop_empty_images)
    )

    # load respective result
    drop = "_drop" if drop_empty_images else ""
    result_coco_path = f"test/test_edit/results/{action}_anno{drop}_coco.json"
    result = cm.COCOParser(result_coco_path)

    # checks
    assert output.annotations.equals(result.annotations)
    assert output.images.equals(result.images)
    assert output.categories.equals(result.categories)


@pytest.mark.parametrize("action", ["remove", "filter"])
def test_remove_filter_img(action: str):
    parser = cm.COCOParser("test/test_edit/fixtures/remove_filter_coco.json")

    # perform remove/filter
    imgs = [1, "0000001.jpg"]
    output = parser.remove_images(imgs) if action == "remove" else parser.filter_images(imgs)

    # load respective result
    result_coco_path = f"test/test_edit/results/{action}_img_coco.json"
    result = cm.COCOParser(result_coco_path)

    # checks
    assert output.annotations.equals(result.annotations)
    assert output.images.equals(result.images)
    assert output.categories.equals(result.categories)


@pytest.mark.parametrize("action", ["remove", "filter"])
@pytest.mark.parametrize("drop_empty_images", [True, False])
def test_remove_filter_cat(action: str, drop_empty_images: bool):
    parser = cm.COCOParser("test/test_edit/fixtures/remove_filter_coco.json")

    # perform remove/filter
    cats = [1, "Van"]
    output = (
        parser.remove_categories(cats, drop_empty_images=drop_empty_images)
        if action == "remove"
        else parser.filter_categories(cats, drop_empty_images=drop_empty_images)
    )

    # load respective result
    drop = "_drop" if drop_empty_images else ""
    result_coco_path = f"test/test_edit/results/{action}_cat{drop}_coco.json"
    result = cm.COCOParser(result_coco_path)

    # checks
    assert output.annotations.equals(result.annotations)
    assert output.images.equals(result.images)
    assert output.categories.equals(result.categories)
