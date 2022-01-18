from pycrostates.utils import get_config


def test_get_config():
    config = get_config()
    assert isinstance(config, dict)
