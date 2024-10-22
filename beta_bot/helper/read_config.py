import json


def read_config():
    with open('./config.json') as f:
        config_data = json.load(f)

    return config_data
