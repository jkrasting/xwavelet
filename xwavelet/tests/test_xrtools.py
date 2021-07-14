import numpy as np
import xarray as xr
from xwavelet.xrtools import (
    isbetween,
    detrend_array,
    infer_time_freq,
    timeseries_stats,
    wavelet,
    _wrap_wavelet,
)
from xwavelet.classes import Wavelet

dset = xr.open_dataset(
    "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc",
    use_cftime=True,
)
dset = dset.isel(time=slice(0, 60))
arr = dset.sst.mean(dim=("lat", "lon"))

import pytest


def pytest_namespace():
    return {"wavelet_1": None}


def test_isbetween():
    valid_range = (10, 20)
    assert isbetween(15, valid_range)
    assert isbetween(16.9, valid_range)
    assert not isbetween(5, valid_range)
    assert not isbetween(20, valid_range)
    assert not isbetween(25, valid_range)


def test_detrend_array():
    result = float(detrend_array(arr).mean())
    assert result != float(arr.mean())
    assert np.allclose(result, 3.1804999923211474e-15)


def test_infer_time_freq():
    assert infer_time_freq(arr) == "M"


def test_timeseries_stats():
    result = timeseries_stats(arr)
    assert np.allclose(result["mean"], 13.477059364318845)
    assert np.allclose(result["stddev"], 0.1581688523292542)
    assert np.allclose(result["arrmin"], 13.1468534469605)
    assert np.allclose(result["arrmax"], 13.80442333221436)
    assert np.allclose(result["skewness"], 0.2928397953510284)
    assert np.allclose(result["range"], 0.657569885253906)


def test_wavelet_1():
    result = wavelet(arr)
    assert isinstance(result, xr.Dataset)
    assert len(result.attrs.keys()) == 9
    assert np.allclose(result.period.sum(), 399.61105775)
    assert np.allclose(result.timeseries.mean(), 3.96719694e-15)
    assert np.allclose(result.spectrum.sum(), 0.55493825)
    assert np.allclose(result.scaled_ts_variance.sum(), 0.00215092)
    assert np.allclose(
        result.scaled_ts_variance.attrs["significance"], 0.015598932038100413
    )
    pytest.wavelet_1 = result


def test_wavelet_2():
    result = wavelet(arr, detrend=False)
    assert np.allclose(result.timeseries.values, arr.values)


def test_wavelet_3():
    result = wavelet(arr, scaled=True)
    assert np.allclose(result.spectrum.sum(), 0.06285915)


def test__wrap_wavelet():
    wave, coi = _wrap_wavelet(arr)
    assert np.allclose(np.abs(wave).sum(), 189.519549900900)
    assert np.allclose(coi.sum(), 614.327108389584)

def test_Wavelet()
    wave = Wavelet(arr)
    assert wave.dset
