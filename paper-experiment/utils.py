import os

def convert_path_to_module_name(config_path):
    return config_path.split('.py')[0].replace('/', '.')


def check_exist_and_mkdir(path):
    if not os.path.exists(path):
        print('make dir', path)
        os.makedirs(path)
