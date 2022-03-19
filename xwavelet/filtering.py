""" filtering.py - tools for filtering a time series in frequency space """

import warnings
import scipy.signal
import numpy as np
import xarray as xr

__all__ = ["frequency_filter"]


def _split_time_string(timestring):
    """Internal function for splitting time cutoff string"""
    tail = timestring.lstrip("0123456789")
    timestring = timestring[::-1]
    head = timestring[len(tail) :]
    head = head[::-1]
    return int(head), tail


def _run_filter(
    arr,
    filter_order,
    cutoff_frequency1,
    cutoff_frequency2,
    btype,
    sampling_frequency,
    iirf_format,
):
    """Internal function for performing the filtering"""

    if str(btype) == "bandpass":
        cutoff_frequency = (
            [cutoff_frequency1, cutoff_frequency2]
            if cutoff_frequency2 is not None
            else cutoff_frequency1
        )
    else:
        cutoff_frequency = cutoff_frequency1

    iirf = scipy.signal.butter(
        filter_order,
        cutoff_frequency,
        btype=str(btype),
        fs=sampling_frequency,
        output=str(iirf_format),
    )

    if iirf_format == "sos":
        filtered = scipy.signal.sosfiltfilt(iirf, arr, axis=-1)

    elif iirf_format == "ba":
        filtered = scipy.signal.filtfilt(*iirf, arr, axis=-1)

    # np.array(filtered)

    return filtered


def frequency_filter(
    arr,
    cutoff,
    dim="time",
    btype="lowpass",
    filter_order=5,
    iirf_format="ba",
    suppress_warnings=False,
):
    """Function to filter an xarray time series

    This function provides an xarray front end to Scipy-based frequency
    filter. A N-th order Butterworth filter, known for its flat frequency
    response in the pass-band, is used in this function and the filter order
    is adjustable at runtime.

    The cutoff frequency is given in the same calendar units as NumPy's
    timedelta class. Some examples are "6M" = 6 months, "1Y" = 1 year,
    "10Y" = 10 years. The function leverages xarray's conversion to
    np.datetime objects for the time coordinate to make this function
    agnostic of the input array's time frequency. See this url for
    more details: https://numpy.org/doc/stable/reference/arrays.datetime.html

    The type of filtering can be either "highpass", "lowpass", or "bandpass"

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input data array
    cutoff : str, or sequence of str
        Time frequency to use as a cutoff. Two values are
        required to use bandpass functionality.
    dim : str, optional
        Name of time coordinate, by default "time"
    btype : str {"lowpass","highpass","bandpass"}, optional
        The type of filter, by default "lowpass"
    filter_order : int
        The order of the filter, by default 5.
    iirf_format : str {"ba","sos"}
        Infinite Impulse Response (IIR) function output format
        numerator/denominator (‘ba’), pole-zero (‘zpk’), or
        second-order sections (‘sos’). Default is ‘ba’, but
        ‘sos’ should be used for general-purpose filtering.
    suppress_warnings : bool
        Turn off warnings, by default False

    Returns
    -------
    xarray.core.dataarray.DataArray
        Filtered version of input data array
    """

    # get the sampling frequency of the data
    time_values = arr[dim]
    sampling_frequency = [
        time_values[x] - time_values[x - 1] for x in range(1, len(time_values))
    ]

    # xarray returns the time coordinate in units of nanoseconds; confirm
    # this is true and convert units to seconds
    try:
        assert (
            str(sampling_frequency[0].values.dtype) == "timedelta64[ns]"
        ), "Time units not in expected type and/or units"
        sampling_frequency = np.array([float(x) * 1.0e-9 for x in sampling_frequency])
    except AssertionError as exception:
        message = (
            "This function expects the time coordinate to be a list "
            + "of `numpy.datetime` objects and that difference between two "
            + "consecutive time steps should be expressed as a `numpy.timedelta64` "
            + "object with units of nanoseconds."
        )
        warnings.warn(message)
        raise exception

    # The sampling frequency should ideally be constant throughout the
    # timeseries. This is not true for monthly data, for example, when
    # the number of days per month varies. In this instance, a mean
    # sampling frequency is used.
    if (len(set(sampling_frequency)) > 1) and suppress_warnings is False:
        warnings.warn("Irregular sampling frequency detected - using an average value.")

    # convert cutoff frequencies to a list if they are not already
    cutoff = [cutoff] if not isinstance(cutoff, list) else cutoff

    # convert the cutoff frequency to seconds
    cutoff_frequency = np.array(
        [
            np.timedelta64(*_split_time_string(x))
            .astype("timedelta64[s]")
            .astype(float)
            for x in cutoff
        ]
    )

    # convert cutoff frequency and sampling frequency from s to Hz
    cutoff_frequency = sorted([1.0 / x for x in cutoff_frequency])
    sampling_frequency = 1.0 / sampling_frequency.mean()

    # separate cutoff frequencies into two parts; xarray connection
    # to dask must pass them independently otherwise it assumes
    # they have specific dimensionality
    cutoff_frequency1 = cutoff_frequency[0]
    cutoff_frequency2 = cutoff_frequency[1] if len(cutoff_frequency) == 2 else None

    # apply the filtering function
    filtered = xr.apply_ufunc(
        _run_filter,
        arr,
        filter_order,
        cutoff_frequency1,
        cutoff_frequency2,
        btype,
        sampling_frequency,
        iirf_format,
        dask="parallelized",
        input_core_dims=[[dim], [], [], [], [], [], []],
        output_core_dims=[[dim]],
        output_dtypes=[arr.dtype],
    )

    return filtered
