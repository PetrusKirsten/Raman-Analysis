"""
Raman Microscopy Analysis Toolkit
===============================

Modular Python pipeline for processing and visualizing Raman spectral maps.
This version is optimized for 2D Raman image creation using bands of interest
and topographic intensity distribution. Designed to be reusable and clean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
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
    parsed = [(int(c.group(1)), int(c.group(2))) for c in coords if c]
    return parsed

# --- Raman Data Loader ---

def load_raman_txt(file_path: str) -> tuple:
    """
    Load Raman spectral map from .txt file into a 3D array.

    :param file_path: Path to .txt file.
    :type file_path: str
    :return: Tuple of (data_cube, raman_shift)
    :rtype: tuple
    """
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    raman_shift = df.iloc[:, 0].values
    spectra_columns = df.columns[1:]
    xy = parse_coordinates(spectra_columns)

    if not xy:
        raise ValueError("No (x/y) coordinates were found. Please check the file delimiter and the format of the column names.")

    max_x = max([x for x, y in xy]) + 1
    max_y = max([y for x, y in xy]) + 1
    data_cube = np.zeros((max_y, max_x, len(raman_shift)))

    for i, (x, y) in enumerate(xy):
        spectrum = df.iloc[:, i + 1].values
        data_cube[y, x, :] = spectrum

    return data_cube, raman_shift

# --- Raman Map Visualizer ---

def plot_raman_map(data_cube: np.ndarray, title: str = "Raman Map - Total Intensity") -> None:
    """
    Plot 2D image of total Raman intensity per pixel.

    :param data_cube: 3D array [y, x, spectrum].
    :type data_cube: np.ndarray
    :param title: Plot title.
    :type title: str
    """
    sum_image = np.sum(data_cube, axis=2)
    sum_image = normalize(correct_outliers(sum_image))

    plt.figure(figsize=(8, 6))
    plt.imshow(sum_image, cmap='inferno')
    plt.title(title)
    plt.colorbar(label='Intensity (a.u.)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

# --- Band Selection Utility ---

def extract_band_map(data_cube: np.ndarray, raman_shift: np.ndarray, center: float, width: float = 10) -> np.ndarray:
    """
    Integrate intensity around a specific Raman band.

    :param data_cube: 3D spectral map.
    :type data_cube: np.ndarray
    :param raman_shift: Spectral axis.
    :type raman_shift: np.ndarray
    :param center: Central wavenumber.
    :type center: float
    :param width: +/- range around center.
    :type width: float
    :return: 2D image of band intensity.
    :rtype: np.ndarray
    """
    min_idx = np.argmin(np.abs(raman_shift - (center - width)))
    max_idx = np.argmin(np.abs(raman_shift - (center + width)))
    band_image = np.sum(data_cube[:, :, min_idx:max_idx + 1], axis=2)
    return normalize(correct_outliers(band_image))
