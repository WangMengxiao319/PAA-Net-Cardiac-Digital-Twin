# Define the data, results, and other local paths that must be configured for the code to work.
# The code will later use the file generated from this code to replace the paths in the code with your own custom ones.
import json


def set_path_mapping(path_mapping_json):
    # parse json:
    # path_dict = json.loads(path_mapping_json)
    # the result is a Python dictionary:
    # print(path_dict["data_path"]) # Test that your path is correct
    with open('../.custom_config/.your_path_mapping.txt', 'w') as f:
        f.write(path_mapping_json)


def get_path_mapping():
    mapping_filename = '../custom_config/your_path_mapping.txt'
    with open(mapping_filename, 'r') as f:
        path_mapping_json = f.read()
    # print(path_mapping_json)
    return json.loads(path_mapping_json)


def get_server_config():
    server_config_filename = '../.custom_config/.your_server_config.txt'
    with open(server_config_filename, 'r') as f:
        server_config_json = f.read()
    return json.loads(server_config_json)


if __name__ == '__main__':
    # path mappings in JSON:
    path_mapping_json = '{"data_path":"/path/to/data/",' \
                        '"results_path":"/path/to/results/"}'
    set_path_mapping(path_mapping_json)
    server_config_json = '{"python_path":"/path/to/python",' \
                         '"code_path":"/path/to/code/"}'

