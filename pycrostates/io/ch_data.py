import numpy as np
from mne.io import Info
from numpy.typing import NDArray

from ..utils._checks import _check_type


class ChData:
    def __init__(self, data: NDArray, info: Info):
        from . import ChInfo  # pylint: disable=import-outside-toplevel

        _check_type(data, (np.ndarray,), "data")
        _check_type(info, (Info,), "info")
        if data.ndim != 2:
            raise ValueError(
                "Argument 'data' should be a 2D array. The "
                f"provided array shape is {data.shape} which has "
                f"{data.ndim} dimensions."
            )
        if not len(info["ch_names"]) == len(data):
            raise ValueError(
                "Instance data and Info do not have the same "
                "number of channels."
            )
        self._data = data
        self._info = ChInfo(info)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        s = f"<{name} | {self._data.shape[-1]} samples >"
        return s

    def _repr_html_(self, caption=None):
        from ..html_templates import (  # pylint: disable=import-outside-toplevel
            repr_templates_env,
        )

        template = repr_templates_env.get_template("ChData.html.jinja")
        info_repr = (
            self._info._repr_html_()
        )  # pylint: disable=protected-access
        return template.render(
            n_samples=self._data.shape[-1], info_repr=info_repr
        )

    @property
    def info(self) -> Info:
        """
        Info.

        :type: `~pycrostates.io.ChInfo`
        """
        return self._info.copy()

    @property
    def data(self) -> NDArray:
        """
        Data.

        :type: `~numpy.array`
        """
        return self._data.copy()
