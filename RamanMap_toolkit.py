"""
Raman Microscopy Analysis Toolkit
===============================

Modular Python pipeline for processing and visualizing Raman spectral maps.
Uses RamanSPy for preprocessing and spectral image handling.
"""

import re
import numpy as np
import pandas as pd
import ramanspy as rp
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.ndimage import median_filter, uniform_filter

# --- Utility Functions ---

def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to range [0, 1].

    :param array: Input array.
    :type array: np.ndarray
    :return: Normalized array.
    :rtype: np.ndarray
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def detect_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Detect outliers using Z-score thresholding.

    :param data: Input array.
    :type data: np.ndarray
    :param threshold: Z-score threshold to identify outliers.
    :type threshold: float
    :return: Boolean mask of outliers.
    :rtype: np.ndarray
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return np.abs(data - mean) > threshold * std

def correct_outliers(array: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Correct outliers in an array using a local filter method.

    :param array: Input array.
    :type array: np.ndarray
    :param method: Filtering method, 'median' or 'mean'.
    :type method: str
    :return: Array with outliers corrected.
    :rtype: np.ndarray
    """
    mask_outliers = detect_outliers(array)
    array_corrected = array.copy()

    if method == 'median':
        corrected_values = median_filter(array, size=3)
    elif method == 'mean':
        corrected_values = uniform_filter(array, size=3)
    else:
        raise ValueError("Invalid method. Choose 'median' or 'mean'.")

    array_corrected[mask_outliers] = corrected_values[mask_outliers]
    return array_corrected

def parse_coordinates(column_names: list) -> list:
    """
    Extract spatial (x, y) coordinates from column names.

    :param column_names: List of column headers.
    :type column_names: list
    :return: List of (x, y) coordinate tuples.
    :rtype: list
    """
    coords = [re.search(r"\((\d+)/(\d+)\)", name) for name in column_names]
    return [(int(c.group(1)), int(c.group(2))) for c in coords if c]

def config_figure(fig_title: str, size: tuple, face: str = 'white', edge: str = '#383838') -> plt.Axes:
    """
    Configure a standard matplotlib figure.

    :param fig_title: Title of the map.
    :type fig_title: str
    :param size: Size in pixels (width, height).
    :type size: tuple
    :param face: Background color.
    :type face: str
    :param edge: Edge color of axes.
    :type edge: str
    :return: Matplotlib axis object.
    :rtype: plt.Axes
    """
    dpi = 300
    height, width = size[0] / dpi, size[1] / dpi

    fig = plt.figure(figsize=(height, width), facecolor=face, edgecolor='w')
    ax = fig.add_subplot(GridSpec(1, 1)[0])
    ax.invert_yaxis()

    ax.set_title(fig_title, color='w')
    ax.tick_params(colors='w')
    ax.set_facecolor(face)

    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)
    ax.spines[['top', 'bottom', 'left', 'right']].set_edgecolor(edge)

    return ax

def config_bar(colorbar) -> None:
    """
    Configure colorbar styling.

    :param colorbar: Matplotlib colorbar object.
    :type colorbar: matplotlib.colorbar.Colorbar
    """
    colorbar.ax.yaxis.set_tick_params(color='w')
    colorbar.ax.tick_params(colors='w')
    colorbar.outline.set_edgecolor('w')
    colorbar.set_label('Intensity', color='w')


# --- Raman Data Loader ---
def load_raman_txt(path: str) -> rp.SpectralImage:
    """
    Load Raman spectral map from .txt file as a RamanSPy SpectralImage.

    :param path: Path to .txt file.
    :type path: str
    :return: SpectralImage containing Raman map.
    :rtype: rp.SpectralImage
    """
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    raman_shift = df.iloc[:, 0].values
    spectra_columns = df.columns[1:]
    xy = parse_coordinates(spectra_columns)

    if not xy:
        raise ValueError("No (x/y) coordinates were found. Please check the file delimiter and the format of the column names.")

    max_x = max(x for x, y in xy) + 1
    max_y = max(y for x, y in xy) + 1
    data_cube = np.zeros((max_y, max_x, len(raman_shift)))

    for i, (x, y) in enumerate(xy):
        spectrum = df.iloc[:, i + 1].values
        data_cube[y, x, :] = spectrum

    return rp.SpectralImage(data_cube, raman_shift)


# --- Raman Preprocessing ---
def preprocess_maps(maps: list, region: tuple, win_len: int) -> list:
    """
    Apply preprocessing pipeline to a list of SpectralImage objects.

    :param maps: List of SpectralImage objects.
    :type maps: list
    :param region: Cropping region (start, end) in wavenumbers.
    :type region: tuple
    :param win_len: Window length for Savitzky-Golay smoothing.
    :type win_len: int
    :return: List of preprocessed SpectralImage objects.
    :rtype: list
    """
    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=region),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
        rp.preprocessing.denoise.SavGol(window_length=win_len, polyorder=3),
        rp.preprocessing.baseline.ASLS(),
    ])
    return [routine.apply(m) if m is not None else None for m in maps]


# --- Intensity Mapping ---
def sum_intensity_map(image: rp.SpectralImage, method: str = 'median') -> np.ndarray:
    """
    Compute total intensity map by summing all shifts.

    :param image: SpectralImage.
    :type image: rp.SpectralImage
    :param method: Outlier correction method.
    :type method: str
    :return: 2D intensity image.
    :rtype: np.ndarray
    """
    total = np.sum(image.spectral_data, axis=2)
    total = correct_outliers(total, method=method)
    return normalize(total)

# --- Raman Map Topography ---
def plot_topography(image: rp.SpectralImage, title: str = "Raman Map - Total Intensity") -> None:
    """
    Plot 2D image of total Raman intensity per pixel.

    :param image: SpectralImage object.
    :type image: rp.SpectralImage
    :param title: Plot title.
    :type title: str
    """
    plt.style.use('seaborn-v0_8-ticks')

    ax = config_figure(title, (2500, 2500), face='#1d1e24', edge='white')

    img = sum_intensity_map(image)

    im = ax.imshow(img, cmap='inferno', origin='lower')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    config_bar(cbar)

    plt.tight_layout()
    plt.show()


# --- Band Selection Utility ---
def extract_band_map(image: rp.SpectralImage,
                     center: float,
                     width: float = 10,
                     method: str = 'median') -> np.ndarray:

    """
    Integrate intensity around a specific Raman band.

    :param image: SpectralImage object.
    :type image: rp.SpectralImage
    :param center: Central wavenumber.
    :type center: float
    :param width: +/- range around center.
    :type width: float
    :return: 2D image of band intensity.
    :rtype: np.ndarray
    """

    shift = image.spectral_axis

    min_idx = np.argmin(np.abs(shift - (center - width)))
    max_idx = np.argmin(np.abs(shift - (center + width)))

    band_image = np.sum(image.spectral_data[:, :, min_idx:max_idx + 1], axis=2)
    band_image = correct_outliers(band_image, method=method)

    return normalize(band_image)

# --- Raman Map Band ---
def plot_band(image: rp.SpectralImage,
              center: float,
              width: float = 10,
              title: str = None,
              figsize: tuple = (2500, 2500),
              cmap: str = 'inferno',
              method: str = 'median'):
    """
    Plot a 2D Raman band intensity map.

    :param image: SpectralImage object.
    :type image: rp.SpectralImage
    :param center: Center wavenumber of the band.
    :type center: float
    :param width: ± range around center to integrate.
    :type width: float
    :param title: Plot title. If None, uses f\"Band {center} cm⁻¹\".
    :type title: str
    :param figsize: Figure size in pixels (width, height).
    :type figsize: tuple
    :param cmap: Colormap name.
    :type cmap: str
    :param method: Outlier correction method ('median' or 'mean').
    :type method: str
    """

    band_img = extract_band_map(image, center, width, method=method)

    fig_title = title or f"Band {center} cm$^{-1}$ Intensity"

    ax = config_figure(fig_title, figsize, face='#1d1e24', edge='white')
    im = ax.imshow(band_img, cmap=cmap, origin='lower')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    config_bar(cbar)

    plt.tight_layout()
    plt.show()

# --- Main Example Execution ---
if __name__ == "__main__":

    # Load raw map
    file_path = "data/St kC CLs/Map St kC CL 14 Region 2.txt"
    raw_map = load_raman_txt(file_path)

    # Preprocess the map
    processed_maps = preprocess_maps([raw_map], region=(250, 1800), win_len=15)
    processed_map = processed_maps[0]

    # Plot total intensity map
    plot_topography(processed_map, title="Total Raman Intensity")

    # Bands to plot
    bands_to_plot = [
        # band, width, title, colormap
        (941, 10, "Starch Band 941 cm$^{-1}$", 'bone'),
        (805, 10, "Iota-Carrageenan 805 cm$^{-1}$", 'bone'),
        (850, 10, "Kappa-Carrageenan 850 cm$^{-1}$", 'bone'),
        (550, 20, "Ca^{2+}$ Interactions 550 cm$^{-1}$", 'bone'),
        (1220, 10, "Kappa S=O 1220 cm$^{-1}$", 'bone'),
    ]

    # Loop to plot each band
    for center, width, title, cmap in bands_to_plot:
        plot_band(
            processed_map,
            center=center,
            width=width,
            title=title,
            cmap=cmap,
            method='median')
