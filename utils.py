import yaml

def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def load_config(file_path):
    yaml.SafeLoader.add_constructor(tag='!join', constructor=join)
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format - {e}")
    return {}