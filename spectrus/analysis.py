# raman/analysis.py

import numpy as np
import ramanspy as rp

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
