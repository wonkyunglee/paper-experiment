

def convert_path_to_module_name(config_path):
    return config_path.split('.py')[0].replace('/', '.')
