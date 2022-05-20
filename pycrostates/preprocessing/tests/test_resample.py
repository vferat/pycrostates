import itertools
from pathlib import Path

import mne
import numpy as np
import pytest
from mne import BaseEpochs
from mne.datasets import testing
from mne.io.pick import _picks_to_idx

from pycrostates.io import ChData
from pycrostates.preprocessing import resample

dir_ = Path(testing.data_path()) / "MEG" / "sample"
fname_raw_testing = dir_ / "sample_audvis_trunc_raw.fif"
raw = mne.io.read_raw_fif(fname_raw_testing, preload=False)
raw = raw.pick("eeg")
raw.load_data()
epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "inst, replace, n_resamples, n_samples",
    [
        (inst, bool_, epo, samp)
        for inst, bool_, epo, samp in itertools.product(
            (raw, epochs),
            (True, False),
            range(1, 11, 2),
            range(100, 700, 200),
        )
    ],
)
def test_resample_n_resamples_n_samples(inst, replace, n_resamples, n_samples):
    """Test resampling with n_resamples and n_samples provided."""
    resamples = resample(
        inst, n_resamples=n_resamples, n_samples=n_samples, replace=replace
    )
    n_ch = _picks_to_idx(inst.info, None, exclude="bads").size
    assert isinstance(resamples, list)
    assert len(resamples) == n_resamples
    for r in resamples:
        assert isinstance(r, ChData)
        assert r._data.shape == (n_ch, n_samples)


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "inst, replace, n_resamples, cov",
    [
        (inst, bool_, epo, samp)
        for inst, bool_, epo, samp in itertools.product(
            (raw, epochs),
            (True, False),
            range(1, 11, 2),
            np.arange(0.2, 0.9, 0.2),
        )
    ],
)
def test_resample_n_resamples_coverage(inst, replace, n_resamples, cov):
    """Test resampling with n_resamples and coverage provided."""
    resamples = resample(
        inst, n_resamples=n_resamples, coverage=cov, replace=replace
    )
    n_ch = _picks_to_idx(inst.info, None, exclude="bads").size
    n_data = inst.times.size
    if isinstance(inst, BaseEpochs):
        n_data *= len(inst)
    assert isinstance(resamples, list)
    assert len(resamples) == n_resamples
    for r in resamples:
        assert isinstance(r, ChData)
        assert r._data.shape == (n_ch, int(cov * n_data / n_resamples))


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "inst, replace, n_samples, cov",
    [
        (inst, bool_, samp, cov)
        for inst, bool_, samp, cov in itertools.product(
            (raw, epochs),
            (True, False),
            range(100, 700, 200),
            np.arange(0.2, 0.9, 0.2),
        )
    ],
)
def test_resample_n_samples_coverage(inst, replace, n_samples, cov):
    """Test resampling with n_samples and coverage provided."""
    resamples = resample(
        inst, n_samples=n_samples, coverage=cov, replace=replace
    )
    n_ch = _picks_to_idx(inst.info, None, exclude="bads").size
    assert isinstance(resamples, list)
    n_data = inst.times.size
    if isinstance(inst, BaseEpochs):
        n_data *= len(inst)
    assert len(resamples) == int(n_data * cov / n_samples)
    for r in resamples:
        assert isinstance(r, ChData)
        assert r._data.shape == (n_ch, n_samples)


def test_resample_raw_noreplace_error():
    """Test error raised when replace is False and not enough samples are
    available."""
    with pytest.raises(ValueError, match="Can not draw 1000 resamples"):
        resample(raw, n_resamples=1000, n_samples=5000, replace=False)
    resample(raw, n_resamples=1000, n_samples=5000, replace=True)


def test_n_resamples_n_samples_coverage_errors():
    """Test error raised by wrong combination of n_resamples, n_samples and
    coverage."""
    # n_resamples is not None
    with pytest.raises(
        ValueError, match="'n_resamples' must be a strictly positive"
    ):
        resample(raw, n_resamples=-1, n_samples=50, replace=False)
    with pytest.raises(ValueError, match="'n_resamples', at least one of"):
        resample(raw, n_resamples=10, replace=False)
    with pytest.raises(ValueError, match="'n_resamples', only one of"):
        resample(
            raw, n_resamples=10, n_samples=50, coverage=0.2, replace=False
        )
    with pytest.raises(ValueError, match="'coverage' must respect"):
        resample(raw, n_resamples=10, coverage=1.2, replace=False)
    with pytest.raises(
        ValueError, match="'n_samples' must be a strictly positive"
    ):
        resample(raw, n_resamples=10, n_samples=-10, replace=False)

    # n_resamples is None
    with pytest.raises(ValueError, match="'n_resamples' is None, both"):
        resample(raw, n_samples=50, replace=False)
    with pytest.raises(ValueError, match="'n_resamples' is None, both"):
        resample(raw, coverage=0.2, replace=False)
    with pytest.raises(
        ValueError, match="'n_samples' must be a strictly positive"
    ):
        resample(raw, n_samples=-10, coverage=0.2, replace=False)
    with pytest.raises(ValueError, match="'coverage' must respect"):
        resample(raw, n_samples=10, coverage=1.2, replace=False)
