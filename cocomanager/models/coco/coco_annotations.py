# Adapted from https://gitlab.com/bifrost-core/forge/-/blob/develop/dataset-utilities/dataset_utilities/models/coco/coco_annotation.py
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    RootModel,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo

NonNegativeNumeric: TypeAlias = Annotated[int, Field(strict=True, ge=0)] | NonNegativeFloat
CocoBboxType: TypeAlias = tuple[NonNegativeNumeric, NonNegativeNumeric, NonNegativeNumeric, NonNegativeNumeric]


class CocoBbox(RootModel[CocoBboxType]):
    def __iter__(self):
        return iter(self.root)

    def as_xywh(self) -> CocoBboxType:
        return self.root

    def as_xyxy(self) -> CocoBboxType:
        x, y, w, h = self.root
        return x, y, x + w, y + h

    def as_xxyy(self) -> CocoBboxType:
        x, y, w, h = self.root
        return x, x + w, y, y + h


class CocoRle(BaseModel):
    counts: list[NonNegativeInt]
    size: tuple[NonNegativeInt, NonNegativeInt]


class CocoPolygon(RootModel[list[NonNegativeNumeric]]):
    def __getitem__(self, item):
        return self.root[item]

    @model_validator(mode="after")
    def has_xy_pairs(self):
        n = len(self.root)
        if n % 2 != 0:
            raise ValueError(f"Length of values in polygon should be even (equal number of x, y points) but found {n}")
        return self

    @property
    def x(self) -> list[NonNegativeNumeric]:
        return self.root[::2]

    @property
    def y(self) -> list[NonNegativeNumeric]:
        return self.root[1::2]


class CocoKeypoints(RootModel[list[NonNegativeNumeric]]):
    class KeypointVisibility(int, Enum):
        NOT_LABELED = 0
        NOT_VISIBLE = 1
        VISIBLE = 2

    model_config = ConfigDict(use_enum_values=True)

    def __getitem__(self, item):
        return self.root[item]

    def __iter__(self):
        return iter(self.root)

    @model_validator(mode="after")
    def is_keypoint_length_valid(self):
        if not hasattr(self, "root") or self.root is None:
            return self
        n = len(self.root)
        if n % 3 != 0:
            raise ValueError(f"Number of keypoints should be a multiple of 3, but is {n}")
        return self

    @model_validator(mode="after")
    def is_keypoint_visibility_valid(self):
        if not hasattr(self, "root") or self.root is None:
            return self
        valid_visiblilities = {member.value for member in self.KeypointVisibility}
        for value in self.root[2::3]:
            if value not in valid_visiblilities:
                raise ValueError(f"Expected keypoint visibility to be 0, 1, or 2 - found {value}")
        return self


class CocoBaseAnnotation(BaseModel):
    id: PositiveInt
    image_id: PositiveInt


class CocoDetectionAnnotation(CocoBaseAnnotation):
    class CrowdType(int, Enum):
        POLYGON = 0
        RLE = 1

    # object detection
    category_id: PositiveInt
    segmentation: CocoRle | list[CocoPolygon]
    area: NonNegativeFloat
    bbox: CocoBbox
    iscrowd: CrowdType

    # keypoint detection
    keypoints: CocoKeypoints | None = None
    num_keypoints: NonNegativeInt | None = None

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("iscrowd")
    @classmethod
    def is_correct_segmentation_format_used(cls, iscrowd: Literal[0, 1], info: ValidationInfo) -> Literal[0, 1]:
        values = info.data
        if "segmentation" not in values:
            return iscrowd
        if iscrowd == cls.CrowdType.POLYGON:
            assert isinstance(
                values["segmentation"], list
            ), f"Polygons should be used if iscrowd={cls.CrowdType.POLYGON}"
            for item in values["segmentation"]:
                assert isinstance(item, CocoPolygon), f"Polygons should be used if iscrowd={cls.CrowdType.POLYGON}"
        elif iscrowd == cls.CrowdType.RLE:
            assert isinstance(values["segmentation"], CocoRle), f"RLE should be used if iscrowd={cls.CrowdType.RLE}"
        return iscrowd

    @field_validator("num_keypoints", mode="before")
    @classmethod
    def dynamically_create_num_keypoints(
        cls, num_keypoints: NonNegativeInt | None, info: ValidationInfo
    ) -> NonNegativeInt | None:
        values = info.data
        keypoints = values.get("keypoints")

        if keypoints is None and num_keypoints is None:
            return None

        if keypoints is None and num_keypoints is not None:
            raise ValueError(f"num_keypoints={num_keypoints} but keypoints is None")

        if keypoints is not None and num_keypoints is None:
            return sum(
                visibility_flag != CocoKeypoints.KeypointVisibility.NOT_LABELED
                for visibility_flag in keypoints.root[2::3]
            )

        return num_keypoints


class CocoCaptionAnnotation(CocoBaseAnnotation):
    caption: str


class CocoAnnotation(BaseModel):
    annotation: CocoDetectionAnnotation | CocoCaptionAnnotation
