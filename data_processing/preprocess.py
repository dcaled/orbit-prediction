# In this part of the challenge you are tasked with obtaining the raw data and processing it to be ready for the
# modelling phase (training and evaluation). In a coding language of your choice (C# or Python are our preferences)
# implement this data processing pipeline, automating it as much as possible. The actual download of the raw files
# does not need to be in code.

import os
import sys
sys.path.insert(0, '..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_config
from setup_logger import logger
from file_parser import FileParser

def list_files(directory):
    try:
        files = [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f))) and \
                 (f.endswith("sp3.gz"))]
        return files
    except FileNotFoundError:
        print("Error: Directory not found.")
        return []


def main():
    # Load configuration
    config = load_config("config.yaml")
    space_object_id = config["space_object"]["id"]
    path_raw_data = config["data"]["path_raw_data"]
    path_clean_data = config["data"]["path_clean_data"]
    skip_lines = config["data"]["skip_lines"]

    files = list_files(path_raw_data)
    n_files = len(files)
    logger.info("File processing start.")
    for i, file in enumerate(files):
        file_path = f"{path_raw_data}/{file}"
        logger.info(f"Processing file [{i + 1}/{n_files}] {file_path} started.")

        FileParser(space_object_id=space_object_id,
                   file_path=file_path,
                   path_clean_data=path_clean_data,
                   skip_lines=skip_lines)

        logger.info(f"Processing file [{i + 1}/{n_files}] {file_path} finished.")
    logger.info(f"File processing finished. {n_files} files parsed.")


if __name__ == '__main__':
    main()
