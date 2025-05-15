import ramanspy as rp

def preprocess(spectrum: rp.Spectrum,
               crop_range: tuple = (40, 1785),
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


def preprocess_batch(spectra_list,
                     crop_range=(380, 1780),
                     smooth_window=7,
                     smooth_polyorder=2):
    """
    Apply the same preprocessing pipeline to a batch of spectra.

    Parameters
    ----------
    spectra_list : list of rp.Spectrum
        List of Spectra to preprocess.
    (outros parâmetros do seu pipeline)

    Returns
    -------
    spectra_processed : list of rp.Spectrum
        List of preprocessed spectra.
    """
    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=crop_range),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=8, threshold=15),
        rp.preprocessing.denoise.SavGol(window_length=smooth_window, polyorder=smooth_polyorder),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.MinMax(),
    ])

    return [routine.apply(spectrum) for spectrum in spectra_list]

