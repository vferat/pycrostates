from _typeshed import Incomplete

def _get_user_dir():
    """Get user directory."""

def _get_home_dir():
    """Get pycrostates config directory."""

def _get_config_path():
    """Get config path."""

def _get_data_path():
    """Get pycrostates data directory."""

default_config: Incomplete

def _save_config(config) -> None:
    """Save pycrostates config."""

def get_config():
    """Read preferences from pycrostates' config file.

    Returns
    -------
    config : dict
        Dictionary containing all preferences as key/values pairs.
    """

def set_config(key, value) -> None:
    """Set preference key in the pycrostates' config file.

    Parameters
    ----------
    key : str
        The preference key to set. Must be a valid key.
    value : str |  None
        The value to assign to the preference key.
    """
