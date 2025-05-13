# raman/plot_utils.py

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from spectrus.analysis import get_peaks


def set_font(font_path: str):

    """
    Set a custom font globally in matplotlib.

    Parameters
    ----------
    font_path : str
        Path to the font file (.ttf or .otf).
    """

    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()

    plt.rcParams.update({

        'axes.facecolor':   'w',
        'figure.facecolor': 'w',
        'axes.edgecolor':   'k',
        'axes.linewidth':   0.75,
        'xtick.color':      'k',
        'ytick.color':      'k',

        'font.family':      font_name,
        'text.color':       'k',
        'axes.labelcolor':  'k',
        'font.size':        16,
        'axes.titlesize':   16,
        'axes.labelsize':   16,
        'xtick.labelsize':  14,
        'ytick.labelsize':  14,
        'legend.fontsize':  14,

        'savefig.dpi':      300,

    })


def config_figure(fig_title: str, size: tuple) -> plt.Axes:

    """
    Create a styled Matplotlib Axes with specified background and edge colors.

    Parameters
    ----------
    fig_title : str
        Title text for the figure.
    size : tuple
        Figure size in pixels (width, height).
    face : str
        Background color.
    edge : str
        Edge color for axes spines.

    Returns
    -------
    ax : plt.Axes
        Configured Matplotlib Axes.
    """
    dpi = 300
    w, h = size[0] / dpi, size[1] / dpi

    fig, ax = plt.subplots(figsize=(w, h))

    ax.set_title(fig_title, weight='bold', pad=12)
    ax.tick_params(direction='out', length=4, width=.75, pad=8)
    ax.set_aspect('auto')   # 'equal' trava, 'auto' é melhor para espectros
    ax.grid(False)

    return ax


def plot_spectrum(spectrum: rp.Spectrum,
                  title: str = "Raman Spectrum",
                  size: tuple = (4500, 2000),
                  color: str = "crimson",
                  linewidth: float = 1.5,
                  highlight_peaks: bool = True,
                  peak_prominence: float = 10):
    """
    Plot a single Spectrum with optional peak highlighting.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Spectrum to plot.
    title : str
        Plot title.
    size : tuple
        Figure size.
    color : str
        Line color.
    linewidth : float
        Line thickness.
    highlight_peaks : bool
        Whether to detect and annotate peaks.
    peak_prominence : float
        Minimum prominence of peaks to detect (if highlight_peaks=True).
    """
    
    ax = config_figure(title, size)

    x = spectrum.spectral_axis
    y = spectrum.spectral_data

    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")

    ax.plot(x, y, color=color, lw=linewidth)

    if highlight_peaks:
        peak_pos, peak_int = get_peaks(spectrum, prominence=peak_prominence)

        ax.plot(peak_pos, peak_int, "ro", label="Peaks")

        for xp, yp in zip(peak_pos, peak_int):
            ax.annotate(f"{xp:.0f}",
                        xy=(xp, yp),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=10,
                        color="red")
    
    plt.tight_layout()
    plt.show()


def plot_stacked(spectra: list,
                 labels: list = None,
                 title: str = "Stacked Raman Spectra",
                 size: tuple = (4500, 2000),
                 linewidth: float = 1.5, transp: float = 0.75,
                 offset_step: float = 1.,
                 highlight_peaks: bool = True,
                 peak_prominence: float = 10):

    """
    Plot stacked Raman spectra with vertical offset.

    Parameters
    ----------
    spectra : list of rp.Spectrum
        List of Spectra to plot.
    labels : list of str, optional
        Labels for legend.
    title : str
        Plot title.
    size : tuple
        Size of figure in pixels.
    linewidth : float
        Thickness of lines.
    offset_step : float
        Vertical offset between spectra (in normalized units).
    highlight_peaks : bool
        If True, mark detected peaks on spectra.
    peak_prominence : float
        Prominence threshold for peak detection.
    """

    ax = config_figure(title, size)

    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")

    for i, spectrum in enumerate(spectra):
       
        x, y = spectrum.spectral_axis, spectrum.spectral_data
        offset = i * offset_step * (np.max(y) - np.min(y))
        label = labels[i] if labels else f"Spectrum {i+1}"

        ax.plot(x, y + offset, lw=linewidth, label=label, color=f"C{i}")

        if highlight_peaks:
            peak_pos, peak_int = get_peaks(spectrum, prominence=peak_prominence)

            ax.plot(peak_pos, peak_int + offset, "ro", markersize=4)

            for xp, yp in zip(peak_pos, peak_int):
                ax.annotate(f"{xp:.0f}",
                            xy=(xp, yp + offset),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                            color="red")

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_area(areas_dict: dict,
                         title: str = "Band Area Comparison",
                         color: str = "deepskyblue",
                         size: tuple = (2000, 1500)):
    
    """
    Plot a bar chart comparing band areas across samples.

    Parameters
    ----------
    areas_dict : dict
        Dictionary with labels as keys and area values.
    title : str
        Plot title.
    color : str
        Bar color.
    size : tuple
        Figure size in pixels.
    """
    
    labels = list(areas_dict.keys())
    values = list(areas_dict.values())

    dpi = 300
    w, h = size[0] / dpi, size[1] / dpi

    fig, ax = plt.subplots(figsize=(w, h))
    ax.bar(labels, values, color=color, edgecolor='black')

    ax.set_title(title, weight='bold', pad=12)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Area")
    ax.tick_params(direction='out', length=4, width=.75, pad=8)

    plt.tight_layout()
    plt.show()