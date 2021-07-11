""" xarray tools for calling wavelet analysis """

import xarray as xr
import xwavelet as xw
import numpy as np
import cftime
import warnings

from scipy.stats import skew


def detrend_array(arr, dim="time", deg=1):
    """Detrend an Xarray DataArray

    Parameters
    ----------
    arr : xarray.DataArray
        Input data array
    dim : str, optional
        Name of time dimension, by default "time"
    deg : int, optional
        Number of polynomial coefficients for fit, by default 1 (linear)

    Returns
    -------
    xarray.DataArray
        Detrended data array
    """
    coeffs = arr.polyfit(dim=dim, deg=deg)
    fitted = xr.polyval(arr[dim], coeffs.polyfit_coefficients)
    return arr - fitted


def power_spectrum(wave, dim="time"):
    """Convert wavelet ouput to a power spectrum

    Parameters
    ----------
    wave : xarray.DataArray
        Data array with dimensions of spectral period and time
    dim : str, optional
        Name of time dimension, by default "time"

    Returns
    -------
    xarray.DataArray
        Time-averaged spectral power
    """
    power = (np.abs(wave)) ** 2
    return power.mean(dim=dim)


def infer_time_freq(arr, dim="time"):
    """Infer dataset time frequency

    Parameters
    ----------
    arr : xarray.DataArray
        Input xarray variable
    dim : str, optional
        Name of time dimension, by default "time"

    Returns
    -------
    str
        Pandas-like string of inferred time frequency
    """
    if arr.time.dt.year[1] == (arr.time.dt.year[0] + 1):
        result = "Y"
    elif arr.time.dt.month[1] == (arr.time.dt.month[0] + 1):
        result = "M"
    elif arr.time.dt.day[1] == (arr.time.dt.day[0] + 1):
        result = "D"
    else:
        result = "unknown"
    return result


def timeseries_stats(arr):
    """Calculate time series statistics

    Parameters
    ----------
    arr : xarray.DataArray
        Input data array

    Returns
    -------
    dict
    """
    return {
        "mean": float(arr.mean()),
        "stddev": float(arr.std()),
        "arrmin": float(arr.min()),
        "arrmax": float(arr.max()),
        "skewness": float(skew(arr.values)),
        "range": float(arr.max().values - arr.min().values),
    }


def wavelet(
    arr,
    dim="time",
    dt=None,
    pad=True,
    dj=0.25,
    pow2=7,
    s0=None,
    mother="MORLET",
    scaled=False,
    detrend=True,
):
    """xarray wrapper for Torrence and Campo wavelet analysis

    Parameters
    ----------
    arr : n-dim xarray.DataArray
        Input timeseries variable
    dim : str, optional
        Name of time dimension, by default "time"
    dt : float, optional
        Time frequency of data, otherwise inferred from time axis, by default None
    pad : bool, optional
        Pad the DataArray, by default True
    dj : float, optional
        Number of suboctaves per octave, by default 0.25 (4 sub-octaves)
    pow2 : int, optional
        Number of power-of-two octaves to analyze, by default 7
    s0 : float, optional
        Starting timescale, otherwise inferred from time axis, by default None
    mother : str, optional
        Wavelet transform of "MORLET", "PAUL", or "DOG", by default "MORLET"
    scaled : boolean, optional
        Rescale the wavelet coefficents so that the double integral is
        the total energy (varaiance multiplied by total time).
        Default is False
    detrend : boolean, optional
        Linearly detrend the input dataset, by default is True

    Returns
    -------
    xarray.DataArray
        n-dim xarray of wavelet transform (period, time)
    """

    # detrend the dataset
    arr = detrend_array(arr, dim=dim) if detrend is True else arr

    # infer time frequency of input array
    dts = {"Y": 1.0, "M": 1.0 / 12.0, "D": 1.0 / 365.0, "unknown": 1.0}
    time_freq = infer_time_freq(arr)
    dt = dts[time_freq]
    if time_freq == "unknown":
        warnings.warn("Unknown time frequency. Using default dt=1.")

    # pad the dataset
    pad = 1 if pad is True else 0

    # starting time scale default values:
    #   1 year for annual data, 6 months for monthly data, 2 weeks for daily data
    default_scale = {"Y": 1.0, "M": 6.0 / 12.0, "D": 14.0 / 365.0, "unknown": 1.0}
    if s0 is None:
        s0 = default_scale[time_freq]

    # calculate powers-of-two with dj sub-octaves each
    j1 = pow2 / dj

    # ufunc does not support unloaded dask arrays
    arr.load()

    # define in/out dimensions
    core_dims = [["time"]]
    output_dims = [["period", "time"], ["time"]]

    # set arguments
    kwargs = {"dt": dt, "pad": pad, "dj": dj, "s0": s0, "J1": j1, "mother": mother}

    # perform wavelet
    result, coi = xr.apply_ufunc(
        _wrap_wavelet,
        arr,
        input_core_dims=core_dims,
        output_core_dims=output_dims,
        kwargs=kwargs,
        vectorize=False,
    )

    # calculate period
    j = np.arange(0, j1 + 1)
    scale = s0 * 2.0 ** (j * dj)
    period = scale

    result = result.assign_coords({"period": period})

    # Rescale the wavelet coefficents so that the double integral
    # is the total energy (varaiance multiplied by total time)
    if scaled is True:

        if mother != "MORLET":
            raise ValueError(
                "Only the MORLET wavelet base can " + "be scaled in this impementation"
            )

        # empirically-derived reconstruction factor
        c_delta = 0.776

        scale = xr.DataArray(scale, dims=("period"))
        result = result * np.sqrt(dt / (c_delta * scale))

    # Convert to time series of spectral power in each period
    power = (np.abs(result)) ** 2

    dset_out = xr.Dataset()
    dset_out["wave"] = result
    dset_out["cone_of_influence"] = xr.DataArray(coi, dims=("time"))
    dset_out["timeseries"] = arr
    dset_out["power"] = power

    return dset_out


def _wrap_wavelet(arr, **kwargs):
    """Thin wrapper for lower level wavelet analysis

    Parameters
    ----------
    arr : xarray.DataArray
        Input data array

    Returns
    -------
    numpy.ndarray
        Results of wavelet transform
    """
    wave, period, scale, coi = xw.wavelets.wavelet(arr, **kwargs)
    return wave, coi
