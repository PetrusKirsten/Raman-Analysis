# raman/utils.py

import numpy as np
import pandas as pd
import ramanspy as rp


def combine_spectra_direct(spectra_list):
    """
    Combine multiple Raman spectra by simple point-wise mean.

    Assumes all spectra have identical Raman Shift arrays.

    Parameters
    ----------
    spectra_list : list of pd.DataFrame
        List of spectra DataFrames to combine.

    Returns
    -------
    combined_spectrum : pd.DataFrame
        Single combined spectrum.

    """
    shifts = [spectrum["Raman Shift (cm-1)"].values for spectrum in spectra_list]
    intensities = [spectrum["Intensity (a.u.)"].values for spectrum in spectra_list]

    # Check if all shifts are equal
    for i in range(1, len(shifts)):
        if not np.allclose(shifts[0], shifts[i], atol=1e-4):
            raise ValueError("Raman Shift arrays are not identical. Check your files.")

    # Mean of intensities
    mean_intensity = np.mean(intensities, axis=0)

    return pd.DataFrame({
        "Raman Shift (cm-1)": shifts[0],
        "Intensity (a.u.)": mean_intensity
    })


def combine_spectra(spectra_list: list) -> rp.Spectrum:

    """
    Combine multiple RamanSPy Spectra by point-wise average.

    Assumes all spectra have identical spectral_axis arrays.

    Parameters
    ----------
    spectra_list : list of rp.Spectrum
        List of RamanSPy Spectra to combine.

    Returns
    -------
    combined_spectrum : rp.Spectrum
        Combined Spectrum object.
    """

    shifts = [spectrum.spectral_axis for spectrum in spectra_list]
    intensities = [spectrum.spectral_data for spectrum in spectra_list]

    # Check all axes are equal
    for i in range(1, len(shifts)):

        if not np.allclose(shifts[0], shifts[i], atol=1e-4):
            raise ValueError("Spectral axes are not identical between spectra.")

    mean_intensity = np.mean(intensities, axis=0)

    return rp.Spectrum(mean_intensity, shifts[0])
