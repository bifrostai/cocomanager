from pydantic import BaseModel, PositiveInt


class CocoCategory(BaseModel):
    id: PositiveInt
    name: str
    supercategory: str | None = None
