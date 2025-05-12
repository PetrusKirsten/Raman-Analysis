# raman/plot_utils.py

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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


def plot_spectra(spectra: list,
                 labels: list = None,
                 title: str = "Raman Spectra Comparison",
                 size: tuple = (4500, 2000),
                 linewidth: float = 1.5):

    """
    Plot multiple RamanSPy spectra on the same plot.

    Parameters
    ----------
    spectra : list of rp.Spectrum
        List of spectra to plot.
    labels : list of str
        List of labels for legend.
    title : str
        Title of the plot.
    size : tuple
        Figure size in pixels.
    linewidth : float
        Thickness of lines.
    """

    ax = config_figure(title, size)

    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")

    for i, spectrum in enumerate(spectra):
        label = labels[i] if labels else f"Spectrum {i+1}"
        ax.plot(
            spectrum.spectral_axis, spectrum.spectral_data,
            color=f"C{i}", lw=linewidth, alpha=0.75,
            label=label)

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_stacked(spectra: list,
                 labels: list = None,
                 title: str = "Stacked Raman Spectra",
                 size: tuple = (4500, 2000),
                 linewidth: float = 1.5, transp: float = 0.75,
                 offset_step: float = 1.):

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
    """

    ax = config_figure(title, size)

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity + offset")

    for i, spectrum in enumerate(spectra):
        x = spectrum.spectral_axis
        y = spectrum.spectral_data
        offset = i * offset_step * (np.max(y) - np.min(y))
        label = labels[i] if labels else f"Spectrum {i+1}"

        ax.plot(x, y + offset,
                lw=linewidth, color=f"C{i}", alpha=transp,
                label=label,)

    ax.legend()
    plt.tight_layout()
    plt.show()