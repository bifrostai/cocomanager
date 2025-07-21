from pydantic import AnyUrl, BaseModel, PositiveInt


class CocoLicense(BaseModel):
    id: PositiveInt
    name: str
    url: AnyUrl
