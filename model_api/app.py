import random
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

app = FastAPI()


class ObjectPositionRequest(BaseModel):
    """
    Input model for requesting satellite position.

    Attributes:
        space_object_id (str): The identifier of the space object.
        number_of_predictions (int): Number of predictions.
    """
    space_object_id: str = Field(..., description="The identifier of the space-object.")
    number_of_predictions: int = Field(10, description="Number of predictions.")

    @field_validator("space_object_id")
    def validate_space_object_id(cls, value):
        allowed_ids = ["larets"]
        if value.lower() not in allowed_ids:
            raise ValueError(f"Invalid space_object_id: {value}. Allowed values: {allowed_ids}")
        return value

class ObjectPosition(BaseModel):
    """
    Output model containing the satellite's position.

    Attributes:
        epoch (datetime): The datetime for which the satellite position is computed.
        x (float): The x-coordinate of the satellite.
        y (float): The y-coordinate of the satellite.
        z (float): The z-coordinate of the satellite.
    """
    epoch: datetime = Field(..., description="The datetime for which the satellite position is computed.")
    x: float = Field(..., description="The x-coordinate of the satellite.")
    y: float = Field(..., description="The y-coordinate of the satellite.")
    z: float = Field(..., description="The z-coordinate of the satellite.")


class ObjectPositionList(BaseModel):
    """
    Output model containing a list of satellite positions.

    Attributes:
        positions (List[ObjectPosition]): A list of satellite positions at different timestamps.
    """
    positions: List[ObjectPosition]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/space-object/position", response_model=ObjectPositionList)
def get_space_object_position(request: ObjectPositionRequest):
    """
    Endpoint to compute satellite position based on input datetime.

    Args:
        request (PositionRequest): Input object containing space object ID and number of predictions.

    Returns:
        ObjectPositionList: List of positions containing epoch, x, y, and z coordinates.
    """
    number_of_predictions = request.number_of_predictions
    positions = []
    for i in range(number_of_predictions):
        positions+=[ObjectPosition(epoch=datetime.now(),
                                   x=random.uniform(1, 2),
                                   y=random.uniform(1, 3),
                                   z=random.uniform(1, 4))]

    return ObjectPositionList(positions=positions)
