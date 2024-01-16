from ._logs import logger as logger

_INSTALL_MAPPING: dict[str, str]

def import_optional_dependency(name: str, extra: str='', raise_error: bool=True):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice message will be
    raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    raise_error : bool
        What to do when a dependency is not found.
        * True : If the module is not installed, raise an ImportError, otherwise, return
                 the module.
        * False: If the module is not installed, issue a warning and return None,
                 otherwise, return the module.

    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module when found.
        None is returned when the package is not found and raise_error is False.
    """