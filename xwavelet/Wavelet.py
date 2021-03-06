import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import xwavelet as xw

class Wavelet:
    def __init__(self, arr, **kwargs):
        self.dset = xw.wavelet(arr, **kwargs)
        self.stats = xw.xrtools.timeseries_stats(self.dset.timeseries)

    def spectrum(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(4.8, 6.4))
            ax = plt.subplot(1, 1, 1)
        logperiod = np.log2(self.dset.period)
        plotarr = self.dset.spectrum
        plotarr = plotarr.assign_coords({"period": logperiod})
        plotarr.plot(ax=ax, y="period")
        ax.invert_yaxis()
        yticks = self.dset.period[0::4]
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels([str(x) for x in yticks.values])

    def density(self, ax=None, cmap="hot_r", add_colorbar=True):
        if ax is None:
            fig = plt.figure(figsize=(8, 4))
            ax = plt.subplot(1, 1, 1)
        logperiod = np.log2(self.dset.period)
        levels = np.log2(self.dset.period).values
        levels = [x / 2.0 for x in levels if x >= 0.0]
        plotarr = np.abs(self.dset.wavelet) ** 2
        plotarr = plotarr.assign_coords({"period": logperiod, "time": plotarr.time})
        plotarr.plot.contourf(
            ax=ax, levels=levels, cmap=cmap, add_colorbar=add_colorbar
        )
        ylabel = ax.get_ylabel()

        levels2 = [x * 2.0 for x in levels if x >= 0.0]
        plotarr.plot.contour(ax=ax, levels=levels2, colors=["k"], linewidths=0.5)

        coi = self.dset.cone_of_influence
        coi = xr.where(coi <= 0, 1e-20, coi)
        coi = np.log2(coi)
        coi.plot.line(
            ax=ax, add_legend=False, linewidth=2, linestyle="dashed", color="k"
        )
        yticks = self.dset.period[0::4]
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels([str(x) for x in yticks.values])
        ax.set_ylim(np.log2(yticks[0]), np.log2(yticks[-1]))
        ax.invert_yaxis()
        ax.set_ylabel(ylabel)

    def timeseries(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 2))
            ax = plt.subplot(1, 1, 1)
        reference = xr.ones_like(self.dset.timeseries) * self.stats["mean"]
        reference.plot.line(ax=ax, linestyle="dashed", color="k")
        self.dset.timeseries.plot.line(ax=ax, linewidth=0.5, color="gray")
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
        annual.plot.line(ax=ax)

    def variance(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 2))
            ax = plt.subplot(1, 1, 1)
        reference = (
            xr.ones_like(self.dset.timeseries)
            * self.dset.scaled_ts_variance.attrs["significance"]
        )
        reference.plot.line(ax=ax, linestyle="dashed", color="k", linewidth=0.5)
        self.dset.scaled_ts_variance.plot.line()

    def composite(self, title=None, subtitle=None):
        fig = plt.figure(figsize=(11, 8.5))
        ax1 = plt.subplot2grid((8, 7), (1, 0), colspan=5, rowspan=2)
        ax2 = plt.subplot2grid((8, 7), (3, 0), colspan=5, rowspan=3)
        ax3 = plt.subplot2grid((8, 7), (3, 5), colspan=2, rowspan=5)
        ax4 = plt.subplot2grid((8, 7), (6, 0), colspan=5, rowspan=2)

        self.timeseries(ax=ax1)
        self.density(ax=ax2, add_colorbar=False)
        self.spectrum(ax=ax3)
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
