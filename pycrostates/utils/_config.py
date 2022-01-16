import os
import json

from pycrostates.preprocessing import resample, extract_gfp_peaks

def _get_home_dir():
    """Get home directory"""
    if os.name.lower() == 'nt':
        parent_dir = os.getenv('USERPROFILE')
    else:
        parent_dir = os.path.expanduser('~')

    home_dir = os.path.join(parent_dir, '.pycrostates')
    if not os.path.isdir(home_dir):
        os.mkdir(home_dir)
    return(home_dir)

def _get_config_path():
    """Get config path"""
    home_dir = _get_home_dir()
    config_path = os.path.join(home_dir, 'pycrostates.json')
    return(config_path)

default_config = {'LEMON_DATASET_PATH' :  os.path.join(_get_home_dir(), 'pycrostates_data', 'LEMON')}

def _save_config(config):
    with open(_get_config_path(), 'w') as f:
        json.dump(config, f)

def get_config():
    config_path = _get_config_path()
    if not os.path.isfile(config_path):
        # create default config
        _save_config(default_config)
    with open(config_path, 'r') as f:
        config =  json.load(f)
    return(config)

def set_config(key, value):
    config = get_config()
    if key in default_config.keys:
        config[key] = value
    _save_config(config)

    