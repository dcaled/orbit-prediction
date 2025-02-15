import yaml


def load_config(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format - {e}")
    return {}