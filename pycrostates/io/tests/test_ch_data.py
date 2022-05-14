import numpy as np
import pytest
from mne import create_info

from pycrostates.io import ChData, ChInfo

# create fake data as 3 sin sources measured across 6 channels
times = np.linspace(0, 5, 2000)
signals = np.array([np.sin(2 * np.pi * k * times) for k in (7, 22, 37)])
coeffs = np.random.rand(6, 3)
data = np.dot(coeffs, signals) + np.random.normal(
    0, 0.1, (coeffs.shape[0], times.size)
)
info = create_info(
    ["Fpz", "Cz", "CPz", "Oz", "M1", "M2"], sfreq=400, ch_types="eeg"
)
ch_info = ChInfo(
    ch_names=["Fpz", "Cz", "CPz", "Oz", "M1", "M2"], ch_types="eeg"
)


def test_ChData():
    """Test basic ChData functionalities."""
    # create from info
    ch_data = ChData(data, info)
    assert np.allclose(ch_data._data, data)
    assert isinstance(ch_data.info, ChInfo)
    assert info.ch_names == ch_data.info.ch_names

    # create from chinfo
    ch_data = ChData(data, ch_info)
    assert np.allclose(ch_data._data, data)
    assert isinstance(ch_data.info, ChInfo)
    assert ch_info.ch_names == ch_data.info.ch_names

    # test ContainsMixin
    assert "eeg" in ch_data
    assert "meg" not in ch_data

    # test MontageMixin
    assert ch_data.get_montage() is None
    ch_data.set_montage("standard_1020")
    assert ch_data.get_montage() is not None
    assert len(ch_data.get_montage().dig) == 9  # 6 channels + 3 fiducials

    # test that data is copied
    data_ = ch_data.get_data()
    data_[0, :] = 0.0
    assert not np.allclose(data_, data)
    assert np.allclose(ch_data._data, data)

    # test get_data() with picks
    data_ = ch_data.get_data(picks="eeg")
    assert np.allclose(ch_data._data, data)
    ch_data.info["bads"] = [ch_data.info["ch_names"][0]]
    data_ = ch_data.get_data(picks="eeg")
    assert np.allclose(ch_data._data, data[1:, :])

    # test repr
    assert isinstance(ch_data.__repr__(), str)
    assert ch_data.__repr__() == f"< ChData | {times.size} samples >"
    assert isinstance(ch_data._repr_html_(), str)

    # test ==
    ch_data1 = ChData(data, info)
    ch_data2 = ChData(data, ch_info)
    assert ch_data1 == ch_data2
    ch_data3 = ChData(data, create_info(6, 400, "eeg"))
    assert ch_data1 != ch_data3
    ch_data3 = ch_data1.copy()
    assert ch_data1 == ch_data3
    ch_data3._dat[0, :] = 0.0
    assert ch_data1 != ch_data3


def test_ChData_invalid_arguments():
    """Test error raised when invalid arguments are provided to ChData."""
    with pytest.raises(
        TypeError, match="'data' must be an instance of ndarray"
    ):
        ChData(list(data[0, :]), create_info(1, 400, "eeg"))
    with pytest.raises(
        TypeError, match="'info' must be an instance of Info or ChInfo"
    ):
        ChData(data, 101)
    with pytest.raises(ValueError, match="'data' should be a 2D array"):
        ChData(data.reshape(6, 5, 400), ch_info)
    with pytest.raises(ValueError, match="'data' and 'info' do not have"):
        ChData(data, create_info(2, 400, "eeg"))
