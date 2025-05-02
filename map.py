import numpy as np
import pandas as pd
import ramanspy as rp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import median_filter, uniform_filter

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

def wnIndex(xAxis: np.ndarray, wavenumber: float) -> int:
    """
    Find the nearest index to a specific wavenumber.

    :param xAxis: Spectral axis.
    :type xAxis: np.ndarray
    :param wavenumber: Target wavenumber.
    :type wavenumber: float
    :return: Index of nearest value.
    :rtype: int
    """
    xAxis = np.asarray(xAxis)
    valid_mask = ~np.isnan(xAxis)
    return np.abs(xAxis[valid_mask] - wavenumber).argmin()

def configFigure(size: tuple, face: str = 'white', edge: str = '#383838') -> plt.Axes:
    """
    Configure a standard matplotlib figure.

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
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)
    ax.spines[['top', 'bottom', 'left', 'right']].set_edgecolor(edge)
    return ax

def drawPeaks(ax: plt.Axes, bands: list):
    """
    Draw vertical bands highlighting peaks.

    :param ax: Matplotlib axis object.
    :type ax: plt.Axes
    :param bands: List of peak positions in wavenumbers.
    :type bands: list
    """
    for band in bands:
        for offset in [-10, 0, 10]:
            ax.axvline(band + offset, color='dimgray', lw=.75, ls='-', alpha=.8, zorder=-1)
        ax.axvline(band, color='whitesmoke', lw=10, ls='-', alpha=.9, zorder=-2)

def readData(file_paths: list) -> list:
    """
    Read spectral data files and convert to RamanSpy SpectralImage objects.

    :param file_paths: List of file paths.
    :type file_paths: list
    :return: List of SpectralImage objects or None.
    :rtype: list
    """
    maps = []
    for filename in file_paths:
        try:
            data = pd.read_csv(filename)
            imageGrid = data['X-Axis']
            spectral_data = data.drop(columns='X-Axis').T.values.reshape(100, 100, 1024)
            maps.append(rp.SpectralImage(spectral_data, imageGrid))
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            maps.append(None)
    return maps

def preprocess(maps: list, region: tuple, winLen: int) -> list:
    """
    Apply preprocessing pipeline to spectral maps.

    :param maps: List of SpectralImage objects.
    :type maps: list
    :param region: Cropping region (start, end) in wavenumbers.
    :type region: tuple
    :param winLen: Window length for Savitzky-Golay smoothing.
    :type winLen: int
    :return: List of preprocessed SpectralImage objects.
    :rtype: list
    """
    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=region),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
        rp.preprocessing.denoise.SavGol(window_length=winLen, polyorder=3),
        rp.preprocessing.baseline.ASLS(),
    ])
    return [routine.apply(m) if m is not None else None for m in maps]

def plotSpectra(imgs: list, peakBands: list, region_crop: tuple, x: int, y: int):
    """
    Plot selected spectra and mean spectra from given pixel coordinates.

    :param imgs: List of SpectralImage objects.
    :type imgs: list
    :param peakBands: List of band positions.
    :type peakBands: list
    :param region_crop: Spectral crop range.
    :type region_crop: tuple
    :param x: X-coordinate of pixel.
    :type x: int
    :param y: Y-coordinate of pixel.
    :type y: int
    """
    axSpec = configFigure((3500, 1100))
    for i, img in enumerate(imgs):
        img[x - 1, y - 1].plot(ax=axSpec, label=f'Region {i + 1} at ({x}, {y})',
                              color=f'C{i}', ls='-', alpha=.75, lw=.85)
        img.mean.plot(ax=axSpec, label=f'Region {i + 1} Mean',
                      color=f'C{i}', ls=':', alpha=.9, lw=1.)
    drawPeaks(axSpec, peakBands)
    axSpec.set_xlim(region_crop)
    axSpec.tick_params(axis='y', which='both', left=False, labelleft=False)
    axSpec.xaxis.set_major_locator(MultipleLocator(100))
    axSpec.xaxis.set_minor_locator(MultipleLocator(25))
    plt.legend()
    plt.tight_layout()

def plotMap(
        imgs: list,
        peakBands: list,
        colors: list,
        region_crop: tuple,
        x: int, y: int,
        save: bool,
        legend: list):
    """
    Plot Raman intensity and topography maps with optional export.

    :param imgs: List of SpectralImage objects.
    :type imgs: list
    :param peakBands: List of peak band positions (wavenumbers).
    :type peakBands: list
    :param colors: List of color maps for visualization.
    :type colors: list
    :param region_crop: Tuple defining the crop region (min, max) for topography.
    :type region_crop: tuple
    :param x: X pixel coordinate for marker.
    :type x: int
    :param y: Y pixel coordinate for marker.
    :type y: int
    :param save: Whether to save plots to disk.
    :type save: bool
    :param legend: List of region labels for plot titles.
    :type legend: list
    """
    def showImage(title: str, data: np.ndarray, cmap: str):
        ax = configFigure((3150, 2450), '#1d1e24', 'w')
        ax.set_facecolor('#1d1e24')
        plt.title(title, color='w', size=14)

        im = ax.imshow(data, cmap=cmap, interpolation='none')
        cbar = plt.colorbar(im, ax=ax)
        cbar.outline.set_edgecolor('w')
        ax.plot(x, y, 'ro', markersize=2, zorder=2)

        ticks = [0, 25, 50, 75, 100]

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticklabels([str(100 - t) for t in ticks])
        ax.tick_params(colors='w')

        plt.tight_layout()
        if save:
            fname = title.replace(' ', '_').replace('-', '_') + '.png'
            plt.savefig(fname, facecolor='#1d1e24', dpi=300)

    for i, img in enumerate(imgs):
        data, wn = img.spectral_data, img.spectral_axis

        for band, cmap in zip(peakBands, colors):
            idx_band = wnIndex(wn, band)
            peak = data[:, :, idx_band]

            b_start, b_end = wnIndex(wn, band - 10), wnIndex(wn, band + 10)
            band_sum = np.sum(data[:, :, b_start:b_end], axis=2)

            topo_start, topo_end = wnIndex(wn, region_crop[0]), wnIndex(wn, region_crop[1])
            topo = np.sum(data[:, :, topo_start:topo_end], axis=2)

            band_corr = normalize(correct_outliers(band_sum))
            topo_corr = normalize(correct_outliers(topo))
            peak_minus_topo = band_corr - topo_corr

            maps = {
                f"{legend[i]} - Band Sum at {band} cm⁻¹": band_corr,
                f"{legend[i]} - Topography": topo_corr,
                f"{legend[i]} - Band minus Topography": peak_minus_topo,
            }

            for title, img_data in maps.items():
                showImage(title, img_data, cmap)

def ramanMicroscopy(
        fileTitle: str, filePath: list,
        regionToCrop: tuple,
        peakBands: list, bandsColor: list,
        plot_map: bool, plot_spectra: bool, save: bool
) -> list:
    """
    Main function to run Raman image processing and visualization.

    :param fileTitle: Title prefix for plots.
    :type fileTitle: str
    :param filePath: List of file paths to spectral data.
    :type filePath: list
    :param regionToCrop: Region (min, max) of spectral axis to crop.
    :type regionToCrop: tuple
    :param peakBands: List of Raman shift values to highlight.
    :type peakBands: list
    :param bandsColor: Color map names for each band.
    :type bandsColor: list
    :param plot_map: Whether to generate Raman maps.
    :type plot_map: bool
    :param plot_spectra: Whether to plot spectral curves.
    :type plot_spectra: bool
    :param save: Whether to save figures to disk.
    :type save: bool
    :return: List of processed SpectralImage objects.
    :rtype: list
    """

    plt.style.use('seaborn-v0_8-ticks')
    legend = [name.split("/")[-1].replace(".txt", "").replace("Map ", "") for name in filePath]
    raw_map = readData(filePath)
    processed_map = preprocess(raw_map, regionToCrop, winLen=16)
    x, y = 50, 50

    if plot_spectra:
        plotSpectra(processed_map, peakBands, regionToCrop, x, y)

    if plot_map:
        plotMap(processed_map, peakBands, bandsColor, regionToCrop, x, y, save, legend)

    return processed_map




if __name__ == '__main__':

    ramanMicroscopy(
        'Topography',
        [
            # "data/Carrageenans/Map 5pct kC Region 1.txt",
            # "data/Carrageenans/Map 5pct iC Region 1.txt",

            # "data/St CLs/Map St CL 0 Region 1.txt",
            # "data/St CLs/Map St CL 0 Region 2.txt",
            # "data/St CLs/Map St CL 7 Region 1.txt",
            # "data/St CLs/Map St CL 7 Region 2.txt",
            # "data/St CLs/Map St CL 14 Region 1.txt",
            # "data/St CLs/Map St CL 14 Region 2.txt",
            # "data/St CLs/Map St CL 21 Region 1.txt",
            # "data/St CLs/Map St CL 21 Region 2.txt",

            # "data/St kC CLs/Map St kC CL 0 Region 1.txt",
            # "data/St kC CLs/Map St kC CL 0 Region 2.txt",
            # "data/St kC CLs/Map St kC CL 7 Region 1.txt",
            # "data/St kC CLs/Map St kC CL 7 Region 2.txt",
            # "data/St kC CLs/Map St kC CL 14 Region 1.txt",
            "data/St kC CLs/Map St kC CL 14 Region 2.txt",
            # "data/St kC CLs/Map St kC CL 21 Region 1.txt",
            # "data/St kC CLs/Map St kC CL 21 Region 2.txt",

            # "data/St iC CLs/Map St iC CL 0 Region 1.txt",
            # "data/St iC CLs/Map St iC CL 0 Region 2.txt",
            # "data/St iC CLs/Map St iC CL 7 Region 1.txt",
            # "data/St iC CLs/Map St iC CL 7 Region 2.txt",
            # "data/St iC CLs/Map St iC CL 14 Region 1.txt",
            # "data/St iC CLs/Map St iC CL 14 Region 2.txt",
            # "data/St iC CLs/Map St iC CL 21 Region 1.txt",
            # "data/St iC CLs/Map St iC CL 21 Region 3.txt",
        ],
        (250, 1800),  # all spectrum: (200, 1800); ideal: (300, 1500)
        [478],  # in wavenumber / Raman shift. Starch principal peak: 478 1/cm 935
        # [805, 850, 941, 1048, 1053, 1220, 1263],  # in wavenumber / Raman shift. Starch principal peak: 478 1/cm
        # ['inferno', 'inferno', 'inferno', 'inferno', 'inferno', 'inferno', 'inferno', ],
        ['inferno',],
        True, True, False)

    plt.show()
