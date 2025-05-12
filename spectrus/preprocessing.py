# raman/preprocessing.py

import ramanspy as rp

def preprocess(spectrum: rp.Spectrum,
               crop_range: tuple = (300, 1785),
               smooth_window: int = 7,
               smooth_polyorder: int = 2) -> rp.Spectrum:
    """
    Apply full preprocessing pipeline to a RamanSPy Spectrum.

    Parameters
    ----------
    spectrum : rp.Spectrum
        RamanSPy Spectrum object.
    crop_range : tuple
        (min_shift, max_shift) range for cropping.
    smooth_window : int
        Window length for Savitzky-Golay smoothing (must be odd).
    smooth_polyorder : int
        Polynomial order for smoothing.

    Returns
    -------
    preprocessed : rp.Spectrum
        Preprocessed Spectrum object.
    """

    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=crop_range),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=8, threshold=15),
        # rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=15),
        rp.preprocessing.denoise.SavGol(window_length=smooth_window, polyorder=smooth_polyorder),
        rp.preprocessing.baseline.ASLS(),
    ])

    return routine.apply(spectrum)
