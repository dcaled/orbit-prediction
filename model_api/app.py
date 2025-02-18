from http.client import HTTPException
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from model_api.inference import OrbitInference

# Initialize inference model
orbit_inference = OrbitInference(config_path="config.yaml")
app = FastAPI()


class ObjectPositionRequest(BaseModel):
    """
    Input model for requesting space-object position.

    Attributes:
        space_object_id (str): The identifier of the space object.
        pos_x (float): The x-coordinate of the space-object.
        pos_y (float): The y-coordinate of the space-object.
        pos_z (float): The z-coordinate of the space-object.
        vel_x (float): The velocity of the space-object in the axis x.
        vel_y (float): The velocity of the space-object in the axis y.
        vel_z (float): The velocity of the space-object in the axis z.
    """
    space_object_id: str = Field(..., description="The identifier of the space-object.")
    epoch: datetime = Field(..., description="The datetime for which the space-object position is computed.")
    pos_x: float = Field(..., description="The x-coordinate of the space-object.")
    pos_y: float = Field(..., description="The y-coordinate of the space-object.")
    pos_z: float = Field(..., description="The z-coordinate of the space-object.")
    vel_x: float = Field(..., description="The velocity of the space-object in the x-axis.")
    vel_y: float = Field(..., description="The velocity of the space-object in the y-axis.")
    vel_z: float = Field(..., description="The velocity of the space-object in the z-axis.")

    @field_validator("space_object_id")
    def validate_space_object_id(cls, value: str) -> str:
        allowed_ids = ["larets"]
        if value.lower() not in allowed_ids:
            raise ValueError(f"Invalid space_object_id: {value}. Allowed values: {allowed_ids}")
        return value

class ObjectPosition(BaseModel):
    """
    Output model containing the space-object's position.

    Attributes:
        epoch (datetime): The datetime for which the space-object position is computed.
        pos_x (float): The x-coordinate of the space-object.
        pos_y (float): The y-coordinate of the space-object.
        pos_z (float): The z-coordinate of the space-object.
    """
    epoch: datetime = Field(..., description="The datetime for which the space-object position is computed.")
    pos_x: float = Field(..., description="The x-coordinate of the space-object.")
    pos_y: float = Field(..., description="The y-coordinate of the space-object.")
    pos_z: float = Field(..., description="The z-coordinate of the space-object.")


class ObjectPositionList(BaseModel):
    """
    Output model containing a list of space-object positions.

    Attributes:
        space_object_id (str): The identifier of the space object.
        positions (List[ObjectPosition]): A list of space-object positions at different timestamps.
    """
    space_object_id: str = Field(..., description="The identifier of the space-object.")
    positions: List[ObjectPosition]


@app.post("/space-object/position", response_model=ObjectPositionList)
def get_space_object_position(request: ObjectPositionRequest):
    """
    Endpoint to compute space-object position based on input datetime.

    Args:
        request (PositionRequest): Input object containing space object ID and number of predictions.

    Returns:
        ObjectPositionList: List of positions containing epoch, x, y, and z coordinates.
    """
    last_position = {
        "epoch": request.epoch.timestamp()* 1000,  # Convert to milliseconds
        "pos_x": request.pos_x,
        "pos_y": request.pos_y,
        "pos_z": request.pos_z,
        "vel_x": request.vel_x,
        "vel_y": request.vel_y,
        "vel_z": request.vel_z
    }

    try:
        predictions = orbit_inference.run_inference(last_position)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    future_positions = []
    for prediction in predictions:
        future_positions+=[
            ObjectPosition(
                epoch=datetime.fromtimestamp(prediction["epoch"] / 1000),
                pos_x=prediction["pos_x"],
                pos_y=prediction["pos_y"],
                pos_z=prediction["pos_z"]
            )
        ]

    return ObjectPositionList(
        space_object_id=request.space_object_id,
        positions=future_positions)
