from pydantic import AnyUrl, BaseModel, PositiveInt


class CocoImages(BaseModel):
    id: PositiveInt
    file_name: str
    height: PositiveInt
    width: PositiveInt

    license: PositiveInt | None = None
    date_captured: str | None = None

    coco_url: AnyUrl | None = None
    flickr_url: AnyUrl | None = None
