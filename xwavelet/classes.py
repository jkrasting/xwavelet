""" classes for xwavelet """

import cftime
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from itertools import cycle
from . import xrtools


class Wavelet:
    """High-level class to run calculation and generate plots"""

    def __init__(self, arr, reference=None, **kwargs):
        """Initializes class by calculating wavelet transform

        Parameters
        ----------
        arr : xarray.DataArray
            Input time series
        """
        timedim = "time"
        self.dset = xrtools.wavelet(arr, **kwargs)
        self.reference = reference
        self.stats = xrtools.timeseries_stats(self.dset.timeseries)
        self.xlim = (
            cftime.date2num(
                self.dset[timedim][0], calendar="noleap", units="days since 2000-01-01"
            ),
            cftime.date2num(
                self.dset[timedim][-1], calendar="noleap", units="days since 2000-01-01"
            ),
        )

    def spectrum(self, ax=None, reference=None):
        """Power spectrum plot

        Parameters
        ----------
        ax : matplotlib.axes.Axes.axis, optional
            Existing matplotlib axis to use for plot, by default None
        reference: xarray.DataArray, list, optional
            Additional specrtra to plot. Must have a period axis.
            Label attribute used in plot legend if present,
            by default None
        """

        reference = self.reference if reference is None else reference

        if ax is None:
            plt.figure(figsize=(4.8, 6.4))
            ax = plt.subplot(1, 1, 1)
        logperiod = np.log2(self.dset.period)
        plotarr = self.dset.spectrum
        plotarr = plotarr.assign_coords({"period": logperiod})
        plotarr.plot(ax=ax, y="period")
        if reference is not None:
            reference = list(reference) if isinstance(reference, tuple) else reference
            reference = [reference] if not isinstance(reference, list) else reference
            lines = ["--", "-.", ":"]
            linecycler = cycle(lines)
        else:
            reference = []

        for x in reference:
            label = x.attrs["label"] if "label" in x.attrs.keys() else None
            logperiod = np.log2(x.period)
            x = x.assign_coords({"period": logperiod})
            x.plot(
                ax=ax,
                y="period",
                linewidth=0.75,
                linestyle=next(linecycler),
                color="gray",
                label=label,
            )

        if len(reference)>0:
            ax.legend(loc=4,fontsize=8)

        ax.invert_yaxis()
        yticks = self.dset.period[0::4]
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels([str(x) for x in yticks.values])
        ax.set_title("")

    def density(self, ax=None, cmap="hot_r", add_colorbar=True):
        """Wavelet density plot

        Parameters
        ----------
        ax : matplotlib.axes.Axes.axis, optional
            Existing matplotlib axis to use for plot, by default None
        cmap : str, optional
            Matplotlib color map, by default "hot_r"
        add_colorbar : bool, optional
            Display colorbar, by default True
        """
        if ax is None:
            plt.figure(figsize=(8, 4))
            ax = plt.subplot(1, 1, 1)
        logperiod = np.log2(self.dset.period)
        levels = np.log2(self.dset.period).values
        levels = [x / 2.0 for x in levels if x >= 0.0]
        plotarr = np.abs(self.dset.wavelet) ** 2
        plotarr = plotarr.assign_coords({"period": logperiod, "time": plotarr.time})
        plotarr.plot.contourf(
            ax=ax,
            levels=levels,
            cmap=cmap,
            add_colorbar=add_colorbar,
            xlim=self.xlim,
        )
        ylabel = ax.get_ylabel()

        levels2 = [x * 2.0 for x in levels if x >= 0.0]
        plotarr.plot.contour(
            ax=ax, levels=levels2, colors=["k"], linewidths=0.5, xlim=self.xlim
        )

        coi = self.dset.cone_of_influence
        coi = xr.where(coi <= 0, 1e-20, coi)
        coi = np.log2(coi)
        coi.plot.line(
            ax=ax,
            add_legend=False,
            linewidth=2,
            linestyle="dashed",
            color="k",
            xlim=self.xlim,
        )
        yticks = self.dset.period[0::4]
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels([str(x) for x in yticks.values])
        ax.set_ylim(np.log2(yticks[0]), np.log2(yticks[-1]))
        ax.invert_yaxis()
        ax.set_ylabel(ylabel)

    def timeseries(self, ax=None, timedim="time"):
        """Time series plot

        Parameters
        ----------
        ax : matplotlib.axes.Axes.axis, optional
            Existing matplotlib axis to use for plot, by default None
        """
        if ax is None:
            plt.figure(figsize=(8, 2))
            ax = plt.subplot(1, 1, 1)
        reference = xr.ones_like(self.dset.timeseries) * self.stats["mean"]
        reference.plot.line(ax=ax, linestyle="dashed", color="k", xlim=self.xlim)
        self.dset.timeseries.plot.line(
            ax=ax, linewidth=0.5, color="gray", xlim=self.xlim
        )
        stats = {k: str(np.round(v, 4)) for k, v in self.stats.items()}
        for n, stat in enumerate(stats.items()):
            ax.text(
                1.04,
                0.9 - (n * 0.15),
                f"{stat[0]} = {stat[1]}",
                ha="left",
                transform=ax.transAxes,
            )
        annual = self.dset.timeseries.rolling({"time": 12}, center=True).mean()
        annual.plot.line(ax=ax, xlim=self.xlim)

    def variance(self, ax=None):
        """Scaled variance plot

        Parameters
        ----------
        ax : matplotlib.axes.Axes.axis, optional
            Existing matplotlib axis to use for plot, by default None
        """
        if ax is None:
            plt.figure(figsize=(8, 2))
            ax = plt.subplot(1, 1, 1)
        reference = (
            xr.ones_like(self.dset.timeseries)
            * self.dset.scaled_ts_variance.attrs["significance"]
        )
        reference.plot.line(
            ax=ax, linestyle="dashed", color="k", linewidth=0.5, xlim=self.xlim
        )
        self.dset.scaled_ts_variance.plot.line()

    def composite(self, title=None, subtitle=None, reference=None):
        """Multi-panel composite plot

        Parameters
        ----------
        title : str, optional
            Main title heading, by default None
        subtitle : str, optional
            Sub title heading, by default None
        reference: xarray.DataArray, list, optional
            Additional specrtra to plot. Must have a period axis.
            Label attribute used in plot legend if present,
            by default None

        Returns
        -------
        matplotlib.figure.Figure
        """

        reference = self.reference if reference is None else reference

        fig = plt.figure(figsize=(11, 8.5))
        ax1 = plt.subplot2grid((8, 7), (1, 0), colspan=5, rowspan=2)
        ax2 = plt.subplot2grid((8, 7), (3, 0), colspan=5, rowspan=3)
        ax3 = plt.subplot2grid((8, 7), (3, 5), colspan=2, rowspan=5)
        ax4 = plt.subplot2grid((8, 7), (6, 0), colspan=5, rowspan=2)

        self.timeseries(ax=ax1)
        self.density(ax=ax2, add_colorbar=False)
        self.spectrum(ax=ax3, reference=reference)
        self.variance(ax=ax4)

        plt.subplots_adjust(hspace=0.5, wspace=1.5)

        ax1.text(
            -0.1,
            1.6,
            title,
            ha="left",
            fontsize=14,
            fontweight="bold",
            transform=ax1.transAxes,
        )
        ax1.text(
            -0.1,
            1.4,
            subtitle,
            ha="left",
            fontsize=12,
            fontstyle="italic",
            transform=ax1.transAxes,
        )

        return fig
