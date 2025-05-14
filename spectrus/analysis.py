import numpy as np
import pandas as pd
import ramanspy as rp
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def extract_band(spectrum: rp.Spectrum, min_shift: float, max_shift: float) -> rp.Spectrum:

    """
    Extract a spectral band (sub-region) from a Spectrum.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Original Spectrum.
    min_shift : float
        Lower bound of the band.
    max_shift : float
        Upper bound of the band.

    Returns
    -------
    band : rp.Spectrum
        Spectrum of the selected band.
    """

    mask = (spectrum.spectral_axis >= min_shift) & (spectrum.spectral_axis <= max_shift)

    return rp.Spectrum(spectrum.spectral_data[mask], spectrum.spectral_axis[mask])


def calculate_peak_area(spectrum: rp.Spectrum, min_shift: float, max_shift: float) -> float:

    """
    Calculate the area under a peak between two Raman shifts.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Spectrum to analyze.
    min_shift : float
        Lower bound of integration.
    max_shift : float
        Upper bound of integration.

    Returns
    -------
    area : float
        Area under the curve.
    """

    mask = (spectrum.spectral_axis >= min_shift) & (spectrum.spectral_axis <= max_shift)

    x = spectrum.spectral_axis[mask]
    y = spectrum.spectral_data[mask]

    return np.trapz(y, x)


def calculate_band_ratio(spectrum: rp.Spectrum,
                         band1_range: tuple,
                         band2_range: tuple) -> float:

    """
    Calculate ratio of areas between two bands.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Spectrum to analyze.
    band1_range : tuple
        (min_shift, max_shift) of band 1 (numerator).
    band2_range : tuple
        (min_shift, max_shift) of band 2 (denominator).

    Returns
    -------
    ratio : float
        Ratio of areas (band1 / band2).
    """

    area1 = calculate_peak_area(spectrum, *band1_range)
    area2 = calculate_peak_area(spectrum, *band2_range)

    if area2 == 0:
        raise ValueError("Band 2 area is zero. Cannot divide.")

    return area1 / area2


def get_peaks(spectrum: rp.Spectrum,
              height: float = None,
              distance: int = None,
              prominence: float = None) -> tuple:
    
    """
    Find peaks in a Raman spectrum.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Spectrum to analyze.
    height : float, optional
        Required height of peaks.
    distance : int, optional
        Minimum horizontal distance (in data points) between peaks.
    prominence : float, optional
        Required prominence of peaks.

    Returns
    -------
    peak_positions : np.ndarray
        Raman Shift positions of the detected peaks.
    peak_intensities : np.ndarray
        Intensities of the detected peaks.
    """
    peaks, props = find_peaks(
        spectrum.spectral_data,
        height=height,
        distance=distance,
        prominence=prominence
    )

    peak_positions = spectrum.spectral_axis[peaks]
    peak_intensities = spectrum.spectral_data[peaks]

    return peak_positions, peak_intensities


def compare_band_areas(
        spectra: list,
        labels: list,
        band_range: tuple) -> dict:

    """
    Calculate and compare band areas across multiple spectra.

    Parameters
    ----------
    spectra : list of rp.Spectrum
        List of spectra.
    labels : list of str
        Corresponding labels for the spectra.
    band_range : tuple
        (min_shift, max_shift) defining the band.

    Returns
    -------
    areas_dict : dict
        Dictionary mapping labels to band areas.
    """
    
    areas = {}

    for spectrum, label in zip(spectra, labels):
        area = calculate_peak_area(spectrum, *band_range)
        areas[label] = area

    return areas

def extract_band_areas(
    spectra: List, 
    labels: List[Tuple[str, float]],
    bands: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Para cada espectro, calcula a área de cada banda definida.

    Parameters
    ----------
    spectra : list of rp.Spectrum
    labels : list of (group, conc)
    bands : dict
        {'nome da banda': (low_shift, high_shift), ...}

    Returns
    -------
    df : pandas.DataFrame
        Colunas: group, conc, cada banda como coluna de área.
    """
    rows = []
    for spec, (group, conc) in zip(spectra, labels):
        row = {'group': group, 'conc': float(conc)}
        for bname, (low, high) in bands.items():
            row[bname] = calculate_peak_area(spec, low, high)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_band_by_formulation(
    df: pd.DataFrame, 
    band: str, 
    out_folder: str = None,
    save: bool = False
):
    """
    Plota área da banda vs concentração para cada group (St, kC, iC).

    Parameters
    ----------
    df : DataFrame de extract_band_areas()
    band : nome da coluna de banda ex. 'C–O–C (480)'
    out_folder : pasta para salvar
    save : se True, salva arquivo .png
    """
    from spectrus.plot_utils import config_figure, addLegend

    ax = config_figure(f"{band}", (4*800, 4*600))
    colors = {
        "St": ['#E1C96B', '#FFE138', '#F1A836', '#E36E34'],
        "St kC": ['hotpink', 'mediumvioletred', '#A251C3', '#773AD1'],
        "St iC": ['lightskyblue', '#62BDC1', '#31A887', '#08653A'],
    }
    for group, grp_df in df.groupby('group'):
        # ordena pelo conc
        grp_df = grp_df.sort_values('conc')
        ax.plot(
            grp_df['conc'], grp_df[band], 
            marker='o', color=colors[group][1], linestyle='-',
            lw=.75, markersize=9, mec=colors[group][1], mfc='w', alpha=1.,
            label=group,
        )

    ax.set_xlabel("CaCl$_2$ concentration (mM)")
    ax.set_ylabel(f"Area under peak")
    ax.set_title(f"{band}" + " cm$^{-1}$")
    ax.set_xticks([0, 7, 14, 21])
    addLegend(ax)
    plt.tight_layout()

    if save and out_folder:
        plt.savefig(f"{out_folder}/band_{band.replace(' ','_')}.png", dpi=300)
    plt.show()


def plot_all_bands(
    df: pd.DataFrame,
    bands: List[str],
    out_folder: str = None,
    save: bool = False
):
    """
    Gera um subplot para cada banda, em uma figura única.
    """
    n = len(bands)
    cols = 2
    rows = (n + 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
    axes = axes.flatten()

    for ax, band in zip(axes, bands):
        for group, grp_df in df.groupby('group'):
            grp_df = grp_df.sort_values('conc')
            ax.plot(
                grp_df['conc'],
                grp_df[band],
                marker='o',
                linestyle='-',
                label=group
            )
        ax.set_title(band)
        ax.set_xlabel("CaCl$_2$ (mM)")
        ax.set_ylabel("Area")
        ax.legend(fontsize=8)
    # remove eixos extras
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    if save and out_folder:
        plt.savefig(f"{out_folder}/all_bands_comparison.png", dpi=300)
    plt.show()