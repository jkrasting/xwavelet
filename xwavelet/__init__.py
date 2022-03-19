""" xwavelet: an xarray front-end for wavelet analysis """

from . import filtering
from . import classes
from . import wavelets
from . import xrtools

frequency_filter = filtering.frequency_filter
infer_time_freq = xrtools.infer_time_freq
power_spectrum = xrtools.power_spectrum
wavelet = xrtools.wavelet
Wavelet = classes.Wavelet
