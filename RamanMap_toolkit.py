"""
Raman Microscopy Analysis Toolkit
================================

Modular Python pipeline for processing and visualizing Raman spectral maps.
Utilizes RamanSPy for preprocessing and spectral image handling.
Includes logging configurations and progress bars for efficient console feedback.
"""

# --------------------------------------
# Imports
# --------------------------------------
import os
import re                                # regular expressions for parsing column names
import numpy as np                       # numerical operations
import pandas as pd                      # data handling
import ramanspy as rp                    # RamanSPy for spectral images and preprocessing
import matplotlib.pyplot as plt          # plotting engine
from sklearn.cluster import KMeans       # K-means clustering
import matplotlib.font_manager as fm     # font settings
from sklearn.decomposition import PCA    # PCA clustering
from matplotlib.gridspec import GridSpec  # figure layout manager
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter  # filters for outlier correction

# Limit the number of physical cores for loky (adjust as needed)
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# print(f'\nList  of availables styles:\n{plt.style.available}\n')
plt.style.use('seaborn-v0_8')

# 1) Adiciona o arquivo de fonte ao FontManager
font_path = ("C:/Users/petru/AppData/Local/Programs/Python/Python313/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/"
             "helvetica-light-587ebe5a59211.ttf")      # ou o seu arquivo FiraSans.otf, etc.
fm.fontManager.addfont(font_path)

# 2) Pega o nome interno da fonte (veja no próprio arquivo ou use fm.FontProperties)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()  # ex: "Helvetica Neue"

# 3) Atualiza o rcParams para usar essa fonte
plt.rcParams.update({
    'font.family':      font_name,
    'font.size':        14,  # default text size
    'axes.titlesize':   16,  # título dos eixos
    'axes.labelsize':   14,  # rótulos X/Y
    'xtick.labelsize':  12,  # números dos ticks
    'ytick.labelsize':  12,
    'legend.fontsize':  12,  # legendas (se usar)
    'text.color':       'whitesmoke',
    'axes.labelcolor':  'whitesmoke',
    'xtick.color':      'whitesmoke',
    'ytick.color':      'whitesmoke',
    'axes.edgecolor':   'k',
    'figure.facecolor': '#09141E',
    'axes.facecolor':   '#09141E',
    'savefig.dpi':      300,
})

# --------------------------------------
# Axis Scaling Utility
# --------------------------------------

POINTS_PER_LINE  = 100      # pixels in X
LINES_PER_IMAGE  = 100      # pixels in Y
SCAN_WIDTH_UM    = 200.0    # µm in X
SCAN_HEIGHT_UM   = 200.0    # µm in Y

def scale_ticks(ax,
                points_x: int = POINTS_PER_LINE,
                lines_y: int = LINES_PER_IMAGE,
                width_um: float = SCAN_WIDTH_UM,
                height_um: float = SCAN_HEIGHT_UM,
                n_ticks: int = 5) -> None:
    """
    Configure X/Y axis ticks by converting pixel indices into micrometers.

    :param ax: Matplotlib Axes to configure.
    :type ax: matplotlib.axes.Axes
    :param points_x: Total number of pixels along the X axis.
    :type points_x: int
    :param lines_y: Total number of pixels along the Y axis.
    :type lines_y: int
    :param width_um: Physical scan width in micrometers (X direction).
    :type width_um: float
    :param height_um: Physical scan height in micrometers (Y direction).
    :type height_um: float
    :param n_ticks: Number of tick marks to display.
    :type n_ticks: int
    """
    import numpy as _np

    # generate equally spaced pixel positions
    x_pix = _np.linspace(0, points_x - 1, n_ticks)
    y_pix = _np.linspace(0, lines_y - 1, n_ticks)

    # convert pixel positions to micrometers
    x_um = _np.linspace(0, width_um, n_ticks)
    y_um = _np.linspace(0, height_um, n_ticks)

    # apply ticks and labels
    ax.set_xticks(x_pix)
    ax.set_xticklabels([f"{x:.0f}" for x in x_um])
    ax.set_yticks(y_pix)
    ax.set_yticklabels([f"{y:.0f}" for y in y_um[::-1]])  # reverse only the ticklabels

    # label axes in micrometers
    ax.set_xlabel("x (µm)", color='whitesmoke', weight='bold')
    ax.set_ylabel("y (µm)", color='whitesmoke', weight='bold')


# --------------------------------------
# Utility Functions
# --------------------------------------

def normalize(array: np.ndarray) -> np.ndarray:
    """
    Scale array values to the [0, 1] range.

    :param array: Input numeric array.
    :type array: np.ndarray
    :return: Normalized array.
    :rtype: np.ndarray
    """

    return (array - np.min(array)) / (np.max(array) - np.min(array))


def normalize_robust(array: np.ndarray,
                     low_pct: float = 2,
                     high_pct: float = 98) -> np.ndarray:
    """
    Scale array to [0,1] using lower/upper percentiles to avoid outlier-driven stretching.
    """
    lo, hi = np.percentile(array, [low_pct, high_pct])
    clipped = np.clip(array, lo, hi)
    return (clipped - lo) / (hi - lo)



def detect_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Identify outliers using Z-score thresholding.

    :param data: Input numeric array.
    :type data: np.ndarray
    :param threshold: Z-score threshold multiplier.
    :type threshold: float
    :return: Boolean mask indicating outliers.
    :rtype: np.ndarray
    """

    mean = np.nanmean(data)
    std = np.nanstd(data)

    return np.abs(data - mean) > threshold * std


def correct_outliers(array: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Replace detected outliers in a 2D array with locally filtered values.

    :param array: Input 2D array.
    :type array: np.ndarray
    :param method: Filtering method, 'median' or 'mean'.
    :type method: str
    :return: Array with outliers corrected.
    :rtype: np.ndarray
    """

    mask = detect_outliers(array)
    corrected = array.copy()

    if method == 'median':
        filtered = median_filter(array, size=3)

    elif method == 'mean':
        filtered = uniform_filter(array, size=3)

    else:
        raise ValueError("Invalid method. Choose 'median' or 'mean'.")

    corrected[mask] = filtered[mask]

    return corrected


def parse_coordinates(column_names: list) -> list:
    """
    Extract (x, y) coordinate tuples from column names formatted as '... (x/y)'.

    :param column_names: List of column header strings.
    :type column_names: list
    :return: List of (x, y) integer tuples.
    :rtype: list
    """

    coords = [re.search(r"\((\d+)/(\d+)\)", name) for name in column_names]

    return [(int(c.group(1)), int(c.group(2))) for c in coords if c]

# --------------------------------------
# Figure Configuration
# --------------------------------------

def config_figure(fig_title: str,
                  size: tuple,
                  face: str = '#09141E',
                  edge: str = 'k') -> plt.Axes:
    """
    Create a styled Matplotlib Axes with specified background and edge colors.

    :param fig_title: Title text for the figure.
    :type fig_title: str
    :param size: Tuple specifying figure size in pixels (width, height).
    :type size: tuple
    :param face: Background color.
    :type face: str
    :param edge: Edge color for axes spines.
    :type edge: str
    :return: Configured Matplotlib Axes.
    :rtype: plt.Axes
    """

    dpi = 300
    w, h = size[0] / dpi, size[1] / dpi

    fig, ax = plt.subplots(figsize=(w, h), facecolor=face)
    ax.set_facecolor(face)

    ax.set_title(fig_title, color='whitesmoke', weight='bold', pad=12)
    ax.tick_params(colors='whitesmoke', direction='out', length=0, width=0, pad=2)
    ax.set_aspect('equal')
    ax.grid(False)  # remover grades de fundo, se usar 'whitegrid' pode manter leves
    for spine in ax.spines.values():
        spine.set_edgecolor(edge)
        spine.set_linewidth(.75)

    return ax


def config_bar(colorbar) -> None:
    """
    Style colorbar ticks and label with white color.

    :param colorbar: Matplotlib Colorbar instance.
    :type colorbar: matplotlib.colorbar.Colorbar
    """

    colorbar.ax.yaxis.set_tick_params(color='whitesmoke')
    colorbar.ax.tick_params(color='whitesmoke', labelcolor='whitesmoke')
    colorbar.outline.set_edgecolor('whitesmoke')
    colorbar.set_label('Normalized intensity', color='whitesmoke', weight='bold', labelpad=8)

# --------------------------------------
# Data Loading
# --------------------------------------

def load_file(path: str) -> rp.SpectralImage:
    """
    Read a .txt file containing a Raman spectral map and return a SpectralImage.

    :param path: File path to the .txt map.
    :type path: str
    :return: SpectralImage object containing 3D data cube and spectral axis.
    :rtype: rp.SpectralImage
    """

    df = pd.read_csv(path, sep=',', encoding='utf-8')
    raman_shift = df.iloc[:, 0].values
    spectra_columns = df.columns[1:]

    xy = parse_coordinates(spectra_columns)
    if not xy:
        raise ValueError("No (x/y) coordinates found; check file format.")

    max_x = max(x for x, y in xy) + 1
    max_y = max(y for x, y in xy) + 1
    data_cube = np.zeros((max_y, max_x, len(raman_shift)))

    for i, (x, y) in enumerate(xy):
        data_cube[y, x, :] = df.iloc[:, i + 1].values

    return rp.SpectralImage(data_cube, raman_shift)

def plot_mean_spectrum(image: rp.SpectralImage,
                       figsize: tuple = (8, 5),
                       title: str = "Mean Raman Spectrum",
                       line_kwargs: dict = None) -> plt.Axes:
    """
    Plot the average Raman spectrum over all pixels of a SpectralImage.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param figsize: Size of the figure in inches (width, height).
    :type figsize: tuple
    :param title: Title of the plot.
    :type title: str
    :param line_kwargs: Optional dict of kwargs to pass to `ax.plot`.
    :type line_kwargs: dict
    :return: The Matplotlib Axes object.
    :rtype: plt.Axes
    """

    # compute mean spectrum
    mean_spec = image.spectral_data.mean(axis=(0, 1))
    wn = image.spectral_axis

    # set up figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    # plot
    kwargs = {} if line_kwargs is None else line_kwargs
    ax.plot(wn, mean_spec, **kwargs)

    # styling
    ax.set_title(title, color='#383838')
    ax.set_xlabel("Raman Shift (cm⁻¹)", color='#383838')
    ax.set_ylabel("Intensity (a.u.)", color='#383838')
    ax.tick_params(colors='#383838', direction='out', length=3, width=.75, pad=3)

    for spine in ax.spines.values():
        spine.set_edgecolor('#383838')
        spine.set_linewidth(.75)
    ax.grid(False)
    plt.tight_layout()

    return ax


# --------------------------------------
# Preprocessing Pipeline
# --------------------------------------

def preprocess(maps: list,
               region: tuple,
               win_len: int) -> list:
    """
    Apply a preprocessing pipeline to SpectralImage objects: crop, despike, smoothing.

    :param maps: List of SpectralImage instances.
    :type maps: list
    :param region: Tuple specifying wavenumber crop range (start, end).
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
        # Optionally add baseline correction, e.g., ASLS
        # rp.preprocessing.baseline.ASLS(),
    ])
    return [routine.apply(m) for m in maps]

# --------------------------------------
# Visualization: Topography
# --------------------------------------

def sum_intensity(image: rp.SpectralImage,
                  method: str = 'mean') -> np.ndarray:
    """
    Sum spectral intensities across all shifts to create a topography map.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param method: Outlier correction method ('median' or 'mean').
    :type method: str
    :return: 2D normalized intensity map.
    :rtype: np.ndarray
    """

    def show_outliers():
        mask = detect_outliers(total, threshold=1.5)
        print("Total pixels:", total.size, "Outliers:", mask.sum())
        plt.figure()
        plt.imshow(mask, cmap='gray', origin='lower')
        plt.title("Outlier Mask")
        plt.show()

    total = np.sum(image.spectral_data, axis=2)
    # show_outliers()
    total = correct_outliers(total, method=method)

    # return normalize(total)
    return normalize_robust(total, low_pct=2, high_pct=98)



def plot_topography(image: rp.SpectralImage,
                    title: str = "Raman Map - Total Intensity") -> None:
    """
    Display the total intensity topography map with styled colorbar.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param title: Figure title.
    :type title: str
    """

    ax = config_figure(title, (2400, 2400))
    topo = sum_intensity(image)

    im = ax.imshow(
        topo,
        cmap='magma',
        interpolation='nearest',
        origin='upper',
        aspect='equal')
    scale_ticks(ax)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    config_bar(cbar)

    plt.tight_layout(pad=0.5)

# --------------------------------------
# Band Extraction and Visualization
# --------------------------------------

def extract_band(image: rp.SpectralImage,
                 center: float,
                 width: float = 10,
                 method: str = 'mean') -> np.ndarray:
    """
    Integrate intensity around a specified Raman shift band.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param center: Center wavenumber of the band.
    :type center: float
    :param width: +/- range around the center shift.
    :type width: float
    :param method: Outlier correction method.
    :type method: str
    :return: 2D normalized band intensity map.
    :rtype: np.ndarray
    """

    shift = image.spectral_axis

    i0 = np.argmin(np.abs(shift - (center - width)))
    i1 = np.argmin(np.abs(shift - (center + width)))

    band = np.sum(image.spectral_data[:, :, i0:i1+1], axis=2)
    band = correct_outliers(band, method=method)

    return normalize(band)


def plot_band(image: rp.SpectralImage,
              center: float,
              width: float = 10,
              title: str = None,
              figsize: tuple = (2500, 2500),
              cmap: str = 'inferno',
              method: str = 'mean',
              compensation: str = 'raw') -> None:
    """
    Plot a single Raman band map, with optional compensation by topography difference.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param center: Center wavenumber.
    :type center: float
    :param width: +/- range around center.
    :type width: float
    :param title: Figure title.
    :type title: str
    :param figsize: Size of figure in pixels.
    :type figsize: tuple
    :param cmap: Colormap for intensity.
    :type cmap: str
    :param method: Outlier correction method.
    :type method: str
    :param compensation: 'raw' for direct band or 'diff' to subtract topography.
    :type compensation: str
    """

    fig_title = title or f"Band {center} cm$^{-1}$ Intensity"
    ax = config_figure(fig_title, figsize, face='#1D1E24', edge='white')

    band_img = extract_band(image, center, width, method=method)

    if compensation == 'diff':
        topo = sum_intensity(image, method=method)
        diff = band_img - topo
        mask = topo > np.percentile(topo, 5)
        diff[~mask] = 0

        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(diff, sigma=0.9)
        clipped = np.clip(smooth, -0.1, +0.1)
        band_img = normalize(clipped)

    im = ax.imshow(normalize(band_img), cmap=cmap, origin='upper')
    scale_ticks(ax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    config_bar(cbar)
    plt.tight_layout()

# --------------------------------------
# Visualization: Multiband RGB
# --------------------------------------

def plot_multiband(image: rp.SpectralImage,
                   bands: list,
                   figsize: tuple = (2500, 2500),
                   method: str = 'mean',
                   colors: list = None,
                   compensation: str = 'raw') -> None:
    """
    Combine multiple single-band maps into a false-color RGB image.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param bands: List of (center, width) tuples for each channel.
    :type bands: list
    :param figsize: Figure size in pixels.
    :type figsize: tuple
    :param method: Outlier correction method.
    :type method: str
    :param colors: List of RGB tuples for each band channel.
    :type colors: list
    :param compensation: 'raw' or 'diff' for topography compensation.
    :type compensation: str
    """

    chan = [extract_band(image, c, w, method=method) for c, w in bands]

    if compensation == 'diff':
        topo = sum_intensity(image, method=method)
        chan = [normalize(np.clip(gaussian_filter(b - topo, sigma=0.9), -0.1, 0.1)) for b in chan]

    if colors is None:
        default = [(1,0,0), (0,1,0), (0,0,1)]
        colors = default[:len(chan)]

    h, w = chan[0].shape
    rgb = np.zeros((h, w, 3))

    for i, band_img in enumerate(chan):
        cr, cg, cb = colors[i]
        rgb[...,0] += band_img * cr
        rgb[...,1] += band_img * cg
        rgb[...,2] += band_img * cb

    rgb = np.clip(rgb, 0, 1)

    ax = config_figure(f"RGB Bands {bands}", figsize, face='#1d1e24', edge='white')
    ax.imshow(rgb, origin='upper')
    scale_ticks(ax)
    plt.axis('off')
    plt.tight_layout()

# --------------------------------------
# Multivariate Analysis: k-means Clustering
# --------------------------------------

def compute_kmeans(image: rp.SpectralImage,
                   n_clusters: int = 4,
                   method: str = 'mean',
                   compensation: str = 'raw') -> np.ndarray:
    """
    Apply k-means clustering to pixel spectra and return a 2D label map.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param n_clusters: Number of clusters.
    :type n_clusters: int
    :param method: Outlier correction method.
    :type method: str
    :param compensation: 'raw' or 'diff' for topography compensation.
    :type compensation: str
    :return: 2D array of cluster labels.
    :rtype: np.ndarray
    """
    BANDS_FOR_CLUSTER = [
        (478, 10),   # starch
        (850, 10),   # kappa-carrageenan
        (1220, 10),  # S=O stretch
        (550, 20),   # Ca2+ interaction
    ]

    h, w, _ = image.spectral_data.shape
    features = []
    topo = sum_intensity(image, method=method) if compensation=='diff' else None

    for center, width in BANDS_FOR_CLUSTER:
        bmap = extract_band(image, center, width, method=method)

        if compensation=='diff':
            diff = bmap - topo
            sm = gaussian_filter(diff, sigma=0.9)

            features.append(normalize(np.clip(sm, -0.1,0.1)).flatten())
        else:
            features.append(bmap.flatten())

    data = np.stack(features, axis=1)

    if compensation=='diff':
        data = data / (data.sum(axis=1, keepdims=True) + 1e-6)

    labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(data)

    return labels.reshape(h, w)

def plot_cluster(labels: np.ndarray,
                 figsize: tuple = (2500, 2500),
                 cmap: str = 'tab10') -> None:
    """
    Plot the cluster label map as a discrete image.

    :param labels: 2D array of cluster labels.
    :type labels: np.ndarray
    :param figsize: Figure size in pixels.
    :type figsize: tuple
    :param cmap: Colormap for clusters.
    :type cmap: str
    """
    ax = config_figure("Cluster Map", figsize, face='#1d1e24', edge='white')
    ax.imshow(labels, cmap=cmap, origin='upper')
    scale_ticks(ax)
    plt.axis('off')
    plt.tight_layout()

# --------------------------------------
# Multivariate Analysis: PCA
# --------------------------------------

def compute_pca(image: rp.SpectralImage,
                n_components: int = 3,
                method: str = 'mean') -> np.ndarray:
    """
    Perform PCA on pixel spectra and return score maps.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param n_components: Number of principal components.
    :type n_components: int
    :param method: 'raw' or 'diff' normalization.
    :type method: str
    :return: Array of shape (h, w, n_components) with PCA scores.
    :rtype: np.ndarray
    """

    h, w, n_shifts = image.spectral_data.shape
    data = image.spectral_data.reshape(-1, n_shifts)

    if method=='diff':
        topo = data.sum(axis=1, keepdims=True)
        data = data / (topo + 1e-6)

    scores = PCA(n_components=n_components, random_state=0).fit_transform(data)

    return scores.reshape(h, w, n_components)

def plot_pca(score_map: np.ndarray,
             component: int = 1,
             figsize: tuple = (2500, 2500),
             cmap: str = 'RdBu_r') -> None:
    """
    Plot the score map for a given principal component.

    :param score_map: 2D array of PCA scores for one component.
    :type score_map: np.ndarray
    :param component: Component index for labeling.
    :type component: int
    :param figsize: Figure size in pixels.
    :type figsize: tuple
    :param cmap: Colormap for the score map.
    :type cmap: str
    """
    ax = config_figure(f"PCA Component {component}", figsize, face='#1d1e24', edge='white')
    im = ax.imshow(score_map, cmap=cmap, origin='upper')
    scale_ticks(ax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    config_bar(cbar)
    plt.tight_layout()

# --------------------------------------
# Main Execution
# --------------------------------------

if __name__ == "__main__":
    print("Run in __main__")
