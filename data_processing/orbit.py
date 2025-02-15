import os
import json
from datetime import datetime
from typing import Dict


class Orbit:
    """
    A class to represent and store an orbit's state at a specific epoch.

    This class encapsulates an object's position and velocity in space
    and provides functionality to persist this data in a structured directory.

    Attributes:
        space_object_id (str): Identifier of the space object.
        epoch (int): The epoch timestamp in milliseconds.
        pos_x (float): X coordinate of the position.
        pos_y (float): Y coordinate of the position.
        pos_z (float): Z coordinate of the position.
        vel_x (float): X component of velocity.
        vel_y (float): Y component of velocity.
        vel_z (float): Z component of velocity.
        path_data (str): Base directory where data will be stored.
    """

    def __init__(
        self,
        space_object_id: str,
        epoch: datetime,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        vel_x: float,
        vel_y: float,
        vel_z: float,
        path_data: str,
    ):
        """
        Initializes an Orbit object.

        Args:
            space_object_id (str): Identifier of the space object.
            epoch (datetime): The epoch timestamp.
            pos_x (float): X position.
            pos_y (float): Y position.
            pos_z (float): Z position.
            vel_x (float): X velocity.
            vel_y (float): Y velocity.
            vel_z (float): Z velocity.
            path_data (str): Base path where data will be stored.
        """
        self.space_object_id = space_object_id
        self.epoch = int(epoch.timestamp()) * 1000  # Convert to milliseconds timestamp
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.path_data = path_data

    def to_dict(self) -> Dict[str, float]:
        """
        Converts the Orbit object into a dictionary.

        Returns:
            Dict[str, float]: A dictionary containing epoch, position, and velocity data.
        """
        return {
            "epoch": self.epoch,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "pos_z": self.pos_z,
            "vel_x": self.vel_x,
            "vel_y": self.vel_y,
            "vel_z": self.vel_z,
        }

    def persist_orbit(self) -> str:
        """
        Saves the orbit's position and velocity data in a structured JSON format.

        The directory structure is:
        {path_data}/{space_object_id}/{epoch}.json

        Returns:
            str: The file path where the data was saved.
        """
        # Ensure directory exists
        os.makedirs(self.path_data, exist_ok=True)

        file_path = os.path.join(self.path_data, f"{self.epoch}.json")

        # Save data to JSON
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

        return file_path
