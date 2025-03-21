import gzip
from typing import List, Tuple

from datetime import datetime
from orbit import Orbit


class FileParser:
    """
    A class for parsing orbit data from a compressed text file.

    Attributes:
        space_object_id (str): Identifier of the space-object.
        file_path (str): The path to the gzip-compressed input file.
        skip_lines (int): The number of lines to skip at the beginning of the file.
        file_chunks (list): A list containing chunks of three lines each from the file.
        orbits (list): A list of Orbit objects parsed from the file.
    """

    def __init__(self, space_object_id: str, file_path: str, path_clean_data: str, skip_lines: int) -> None:
        """
        Initializes the FileParser object and processes the file.

        Args:
            space_object_id (str): Identifier of the space-object.
            file_path (str): The path to the gzip-compressed input file.
            skip_lines (int): The number of lines to skip at the beginning of the file.
        """

        self.space_object_id = space_object_id
        self.file_path = file_path
        self.path_clean_data = path_clean_data
        self.skip_lines = skip_lines
        self.file_chunks = self.read_file_in_chunks()
        self.orbits = self.parse_chunks()

    def read_file_in_chunks(self, chunk_size: int = 3) -> List[List[bytes]]:
        """
        Reads the file and splits it into chunks of three lines each.

        Args:
            chunk_size (int, optional): The number of lines per chunk. Defaults to 3.

        Returns:
            list: A list of chunks, where each chunk is a list of three lines.
        """
        chunks = []
        with gzip.open(self.file_path, 'rb') as file:
            # Skip the first specified number of lines
            for _ in range(self.skip_lines):
                next(file, None)

            # Read the file in chunks of 3 lines
            while True:
                lines = [file.readline().strip() for _ in range(chunk_size)]
                # Stop when the end of file is reached
                if lines == [b'EOF', b'', b'']:
                    break
                chunks+=[lines]
        return chunks

    def parse_chunks(self):
        """
        Parses all chunks into Orbit objects.

        Returns:
            list: A list of Orbit objects.
        """
        for chunk in self.file_chunks:
            orbit = self.parse_chunk(chunk)
            orbit.persist_orbit()
        return

    def parse_chunk(self, chunk: List[bytes]) -> Orbit:
        """
        Parses a single chunk into an Orbit object.

        Args:
            chunk (list): A list of three lines representing epoch, position, and velocity.

        Returns:
            Orbit: An Orbit object containing the parsed data.
        """
        epoch = self.convert_datetime_format(chunk[0].decode("utf-8"))
        x, y, z = self.parse_position(chunk[1].decode("utf-8"))
        vx, vy, vz = self.parse_velocity(chunk[2].decode("utf-8"))

        orbit = Orbit(space_object_id=self.space_object_id,
                      epoch=epoch,
                      pos_x=x,
                      pos_y=y,
                      pos_z=z,
                      vel_x=vx,
                      vel_y=vy,
                      vel_z=vz,
                      path_data=self.path_clean_data)
        # logger.info(orbit.__dict__)
        return orbit

    @staticmethod
    def convert_datetime_format(date_string: str) -> datetime:
        """
        Converts a date string into a datetime object.

        Args:
            date_string (str): A string representing the date in the format "*  YYYY MM DD HH MM SS.SSSSSSSS".

        Returns:
            datetime: A datetime object representing the given date.
        """
        parts = date_string.replace("*  ", "").split()
        year, month, day, hour, minute = map(int, parts[:5])
        second, microsecond = map(int, parts[5].split("."))
        return datetime(year, month, day, hour, minute, second, microsecond)

    @staticmethod
    def parse_position(position_string: str) -> Tuple[float, float, float]:
        """
        Parses a position string into x, y, and z coordinates.

        Args:
            position_string (str): A string containing three floating-point values representing position.

        Returns:
            tuple: A tuple (x, y, z) of floats representing the position.
        """
        parts = position_string.replace("PL59", "").split()
        x, y, z = map(float, parts)
        return x, y, z

    @staticmethod
    def parse_velocity(velocity_string: str) -> Tuple[float, float, float]:
        """
        Parses a velocity string into vx, vy, and vz components.

        Args:
            velocity_string (str): A string containing three floating-point values representing velocity.

        Returns:
            tuple: A tuple (vx, vy, vz) of floats representing the velocity.
        """
        parts = velocity_string.replace("VL59", "").split()
        vx, vy, vz = map(float, parts)
        return vx, vy, vz
