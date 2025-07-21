from typing import Annotated

from pydantic import AnyUrl, BaseModel, Field


class CocoInfo(BaseModel):
    year: Annotated[int, Field(strict=True, ge=0)]
    version: str
    description: str
    contributor: str
    url: AnyUrl
    date_created: str  # TODO: validate datetime format
