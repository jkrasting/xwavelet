""" setup for xwavelet """
from setuptools import setup, find_packages
import os

setup(
    name="xwavelet",
    version="0.0.1",
    author="John Krasting",
    author_email="John.Krasting@noaa.gov",
    description=("xarray-based implementation of wavelet analysis"),
    license="MIT",
    keywords="",
    url="https://github.com/jkrasting/xwavelet",
    packages=find_packages(),
)
