import json
import os


def _get_user_dir():
    """Get user directory."""
    if os.name.lower() == "nt":
        user_dir = os.getenv("USERPROFILE")
    else:
        user_dir = os.path.expanduser("~")
    return user_dir


def _get_home_dir():
    """Get pycrostates config directory."""
    user_dir = _get_user_dir()
    home_dir = os.path.join(user_dir, ".pycrostates")
    if not os.path.isdir(home_dir):
        os.mkdir(home_dir)
    return home_dir


def _get_config_path():
    """Get config path."""
    home_dir = _get_home_dir()
    config_path = os.path.join(home_dir, "pycrostates.json")
    return config_path


def _get_data_path():
    """Get pycrostates data directory."""
    user_dir = _get_user_dir()
    data_dir = os.path.join(user_dir, "pycrostates_data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    return data_dir


default_config = {
    "PREPROCESSED_LEMON_DATASET_PATH": os.path.join(
        _get_data_path(), "PREPROCESSED_LEMON"
    )
}


def _save_config(config):
    """Save pycrostates config."""
    with open(_get_config_path(), "w", encoding="utf-8") as f:
        json.dump(config, f)


def get_config():
    """Read preferences from pycrostates' config file.

    Returns
    -------
    config : dict
        Dictionary containing all preferences as key/values pairs.
    """
    config_path = _get_config_path()
    if not os.path.isfile(config_path):
        # create default config
        _save_config(default_config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def set_config(key, value):
    """Set preference key in the pycrostates' config file.

    Parameters
    ----------
    key : str
        The preference key to set. Must be a valid key.
    value : str |  None
        The value to assign to the preference key.
    """
    config = get_config()
    if key in default_config:
        config[key] = value
    else:
        raise ValueError("Invalid key")
    _save_config(config)
