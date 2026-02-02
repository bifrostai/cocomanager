from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator


class UpdateAnnotationModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        frozen=True,
    )

    possible_category_ids: list[NonNegativeInt] = Field(..., min_length=1)
    bbox: list[NonNegativeInt] | None = Field(None, min_length=4, max_length=4)
    category_id: NonNegativeInt | None = None

    @field_validator("category_id")
    @classmethod
    def check_category_id(cls, category_id, info):
        values = info.data
        possible_cat = values["possible_category_ids"]
        if category_id is not None and category_id not in possible_cat:
            raise ValueError(f"Category {category_id} is not within category list: {possible_cat}")
        return category_id
