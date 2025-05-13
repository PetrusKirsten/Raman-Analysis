import pandas as pd
import ramanspy as rp

def load_spectrum(file_path: str) -> rp.Spectrum:
    """
    Load a txt Raman spectrum and return as ramanspy Spectrum object.

    Parameters
    ----------
    file_path : str
        Path to the txt file.

    Returns
    -------
    spectrum : rp.Spectrum
        RamanSPy Spectrum object.
    """

    # Read txt file (skip header)
    df = pd.read_csv(
        file_path,
        skiprows=1,
        header=None,
        names=["Raman Shift (cm-1)", "Intensity (a.u.)"],
        sep=",\s+",
        engine="python"
    )

    shifts = df["Raman Shift (cm-1)"].values
    intensity = df["Intensity (a.u.)"].values

    return rp.Spectrum(intensity, shifts)
