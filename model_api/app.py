from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime

app = FastAPI()


class PositionRequest(BaseModel):
    """
    Input model for requesting satellite position.

    Attributes:
        timestamp (datetime): The datetime for which the satellite position is requested.
    """
    timestamp: datetime = Field(..., description="The datetime for which the satellite position is requested.")


class PositionResponse(BaseModel):
    """
    Output model containing the satellite's position.

    Attributes:
        x (float): The x-coordinate of the satellite.
        y (float): The y-coordinate of the satellite.
        z (float): The z-coordinate of the satellite.
    """
    x: float = Field(..., description="The x-coordinate of the satellite.")
    y: float = Field(..., description="The y-coordinate of the satellite.")
    z: float = Field(..., description="The z-coordinate of the satellite.")



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/satellite/position", response_model=PositionResponse)
def get_satellite_position(request: PositionRequest):
    """
    Endpoint to compute satellite position based on input datetime.

    Args:
        request (PositionRequest): Input object containing timestamp.

    Returns:
        PositionResponse: Object containing x, y, and z coordinates.
    """
    # Dummy logic to generate x, y, z values based on timestamp
    timestamp_seconds = request.timestamp.timestamp()  # Convert datetime to seconds
    x = timestamp_seconds % 10000
    y = (timestamp_seconds / 2) % 10000
    z = (timestamp_seconds / 3) % 10000

    return PositionResponse(x=x, y=y, z=z)