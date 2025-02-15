import os
import json


class Orbit:
    def __init__(self, space_object_id, epoch, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, path_data):
        """
        Initialize the Orbit object.

        Args:
            space_object_id (str): Identifier of the space-object.
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

    def persist_orbit(self):
        """
        Save the object's position and velocity data at a given timestamp.

        Data is stored in a structured directory:
        {path_data}/{space_object_id}/{epoch}.json

        The saved data contains position and velocity vectors.
        """
        # Define the directory path
        object_dir = os.path.join(self.path_data, self.space_object_id)
        os.makedirs(object_dir, exist_ok=True)  # Ensure directory exists

        # Define the file path
        file_path = os.path.join(object_dir, f"{self.epoch}.json")

        # Create a dictionary with position and velocity data
        orbit_data = self.__dict__

        # Save the data as a JSON file
        with open(file_path, "w") as f:
            json.dump(orbit_data, f, indent=4)

        print(f"Orbit data saved to {file_path}")
