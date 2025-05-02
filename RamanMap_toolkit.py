"""
Raman Microscopy Analysis Toolkit
================================

Modular Python pipeline for processing and visualizing Raman spectral maps.
Uses RamanSPy for preprocessing and spectral image handling.
Includes logging and progress bars for efficient console feedback.
"""

# --------------------------------------
# Imports
# --------------------------------------
import os
import re                                                # regex for parsing column names
import logging                                           # logging and progress bar
import numpy as np                                       # numerical operations
import pandas as pd                                      # data handling
from tqdm import tqdm                                    # progress bar
import ramanspy as rp                                    # RamanSPy for spectral images and preprocessing
import matplotlib.pyplot as plt                          # plotting
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec                 # figure layout
from scipy.ndimage import median_filter, uniform_filter  # outlier correction filters

# limite o loky a usar 1 core físico (pode ajustar para o nº de cores que você tem)
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# --------------------------------------
# Logging Configuration
# --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------
# Utility Functions
# --------------------------------------

def normalize(array: np.ndarray) -> np.ndarray:
    """
    Scale array values to the [0, 1] range.
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def detect_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Identify outliers via Z-score thresholding.
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return np.abs(data - mean) > threshold * std


def correct_outliers(array: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Replace outliers in a 2D map with local filtered values.
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
    Extract (x, y) integer tuples from column headers like '... (x/y)'.
    """
    coords = [re.search(r"\((\d+)/(\d+)\)", name) for name in column_names]
    return [(int(c.group(1)), int(c.group(2))) for c in coords if c]


# --------------------------------------
# Figure Configuration
# --------------------------------------

def config_figure(fig_title: str,
                  size: tuple,
                  face: str = 'white',
                  edge: str = '#383838') -> plt.Axes:
    """
    Create a styled matplotlib Axes with dark/light background.
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
    Style colorbar ticks and label in white.
    """
    colorbar.ax.yaxis.set_tick_params(color='w')
    colorbar.ax.tick_params(colors='w')
    colorbar.outline.set_edgecolor('w')
    colorbar.set_label('Intensity', color='w')


# --------------------------------------
# Data Loading
# --------------------------------------

def load_file(path: str) -> rp.SpectralImage:
    """
    Read a .txt Raman map and return a RamanSPy SpectralImage.
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


# --------------------------------------
# Preprocessing Pipeline
# --------------------------------------

def preprocess(maps: list,
               region: tuple,
               win_len: int) -> list:
    """
    Crop, despike, smooth, and baseline-correct SpectralImage objects.
    """
    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=region),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
        rp.preprocessing.denoise.SavGol(window_length=win_len, polyorder=3),
        rp.preprocessing.baseline.ASLS(),
    ])
    return [routine.apply(m) for m in maps]


# --------------------------------------
# Visualization: Topography
# --------------------------------------

def sum_intensity(image: rp.SpectralImage,
                  method: str = 'median') -> np.ndarray:
    """
    Sum all spectral intensities per pixel to create a topography map.
    """
    total = np.sum(image.spectral_data, axis=2)
    total = correct_outliers(total, method=method)
    return normalize(total)


def plot_topography(image: rp.SpectralImage,
                    title: str = "Raman Map - Total Intensity") -> None:
    """
    Display the total intensity map with styled colorbar.
    """
    plt.style.use('seaborn-v0_8-ticks')
    ax = config_figure(title, (2500, 2500), face='#1d1e24', edge='white')
    img = sum_intensity(image)
    im = ax.imshow(img, cmap='inferno', origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    config_bar(cbar)
    plt.tight_layout()
    plt.show()


# --------------------------------------
# Band Extraction and Visualization
# --------------------------------------

def extract_band(image: rp.SpectralImage,
                 center: float,
                 width: float = 10,
                 method: str = 'median') -> np.ndarray:
    """
    Integrate intensity around a given Raman shift window.
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
              method: str = 'median',
              compensation: str = 'raw') -> None:
    """
    Plot a single-band map, optionally compensating by topography difference.
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
    else:
        band_img = normalize(band_img)
    im = ax.imshow(band_img, cmap=cmap, origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    config_bar(cbar)
    plt.tight_layout()


# --------------------------------------
# Visualization: Multiband RGB
# --------------------------------------

def plot_multiband(image: rp.SpectralImage,
                   bands: list,
                   figsize: tuple = (2500, 2500),
                   method: str = 'median',
                   colors: list = None,
                   compensation: str = 'raw'):

    """
    Combine three single-band maps into an RGB image.
    """

    # 1) extrai cada canal
    chan = [extract_band(image, c, w, method=method) for c, w in bands]

    # Se for modo 'diff', subtrai topografia de cada banda
    if compensation == 'diff':
        topo = sum_intensity(image, method=method)

        chan_diff = []

        for band_img in chan:
            diff = band_img - topo

            # máscara de confiabilidade
            mask = topo > np.percentile(topo, 5)
            diff[~mask] = 0

            # suaviza e clipe
            from scipy.ndimage import gaussian_filter
            smooth = gaussian_filter(diff, sigma=0.9)
            clipped = np.clip(smooth, -0.1, +0.1)
            chan_diff.append(normalize(clipped))

        chan = chan_diff

    # define cores padrão (R,G,B) se usuário não passou
    if colors is None:
        # pra 2 bandas: vermelho e verde; pra 3: R,G,B
        default = [(1,0,0), (0,1,0), (0,0,1)]
        colors = default[:len(chan)]

    # cria rgb vazio
    h, w = chan[0].shape
    rgb = np.zeros((h, w, 3))

    # mistura cada banda em seu canal, segundo a cor desejada
    for i, band_img in enumerate(chan):
        cr, cg, cb = colors[i]
        rgb[..., 0] += band_img * cr
        rgb[..., 1] += band_img * cg
        rgb[..., 2] += band_img * cb
    rgb = np.clip(rgb, 0, 1)

    ax = config_figure(f"RGB Bands {bands}", figsize, face='#1d1e24', edge='white')
    ax.imshow(rgb, origin='lower')

    plt.axis('off')
    plt.tight_layout()


# --------------------------------------
# Multivariate Analysis: k-means Clustering
# --------------------------------------

def compute_kmeans(image: rp.SpectralImage,
                   n_clusters: int = 4,
                   method: str = 'median',
                   compensation: str = 'raw') -> np.ndarray:
    """
    Aplica k-means aos espectros de cada pixel e retorna
    um mapa 2D com rótulos de cluster.
    """

    # Bandas químicas para clustering
    BANDS_FOR_CLUSTER = [
        (478, 10),  # amido
        (850, 10),  # κ-carragenana
        (1220, 10),  # S=O de carragenana
        (550, 20),  # interações Ca2+
    ]

    # 1) extrai features químicas: banda normalizada ou compensada
    h, w, _ = image.spectral_data.shape
    features = []
    topo = None
    if compensation == 'diff':
        # só calcula topo uma vez
        topo = sum_intensity(image, method=method)

    for center, width in BANDS_FOR_CLUSTER:
        band_map = extract_band(image, center, width, method=method)

        if compensation == 'diff':
            # subtrai topo e normaliza diferença
            diff = band_map - topo

            from scipy.ndimage import gaussian_filter
            smooth = gaussian_filter(diff, sigma=0.9)
            clipped = np.clip(smooth, -0.1, +0.1)
            feat = normalize(clipped)

        else:
            feat = band_map

        features.append(feat.flatten())

    # 2) empilha as colunas: shape (h*w, n_bands)
    data = np.stack(features, axis=1)

    # 2) compensação opcional: normaliza cada espectro pela intensidade total
    if compensation == 'diff':
        total = data.sum(axis=1, keepdims=True)
        # evita divisão por zero
        data = data / (total + 1e-6)

    # 3) roda o k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_.reshape(h, w)

    return labels

def plot_cluster(labels: np.ndarray,
                 figsize: tuple = (2500, 2500),
                 cmap: str = 'tab10') -> None:
    """
    Plota o mapa de clusters como imagem discreta.
    """
    ax = config_figure("Cluster Map", figsize, face='#1d1e24', edge='white')
    im = ax.imshow(labels, cmap=cmap, origin='lower')

    plt.axis('off')
    plt.tight_layout()


# --------------------------------------
# Multivariate Analysis: PCA
# --------------------------------------
def compute_pca(image: rp.SpectralImage,
                n_components: int = 3,
                method: str = 'median') -> np.ndarray:
    """
    Executa PCA nos espectros de cada pixel.
    Retorna um array (h, w, n_components) com os scores.
    """

    # 1) extrair cubo h×w×n_shifts e remodelar em (n_pixels, n_shifts)
    h, w, n_shifts = image.spectral_data.shape
    data = image.spectral_data.reshape(-1, n_shifts)

    # 2) opcional: normalizar espectros pelo total (tal como no k-means diff)
    if method == 'diff':
        topo = data.sum(axis=1, keepdims=True)
        data = data / (topo + 1e-6)

    # 3) rodar o PCA
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(data)  # shape (n_pixels, n_components)

    # 4) remodelar de volta para (h, w, n_components)
    return scores.reshape(h, w, n_components)


def plot_pca(score_map: np.ndarray,
             component: int = 1,
             figsize: tuple = (2500, 2500),
             cmap: str = 'RdBu_r') -> None:
    """
    Plota o mapa de scores de um componente principal.
    """
    ax = config_figure(f"PCA Component {component}", figsize, face='#1d1e24', edge='white')
    im = ax.imshow(score_map, cmap=cmap, origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    config_bar(cbar)
    plt.tight_layout()


# --------------------------------------
# Main Execution
# --------------------------------------

if __name__ == "__main__":

    def showBands():

        # Bands to plot: (center, width, title, cmap, compensation)
        bands_to_plot = [
            (941, 10, "Starch Band 941 cm$^{-1}$", 'bone', 'diff'),
        ]
        # Loop with progress bar
        for center, width, title, cmap, comp in tqdm(
                bands_to_plot, desc="Plotting bands", unit="band"):
            logger.info(f"{title} ({comp})...")

            plot_band(
                processed_map,
                center=center,
                width=width,
                title=title,
                cmap=cmap,
                method='median',
                compensation=comp
            )

    def showMultibands():

        logger.info("Plotting multiband map...")
        plot_multiband(
            processed_map,
            bands=[(941, 10), (850, 10)],
            colors=[(1, 1, 0), (0, 0, 1)],
            method='median',
            compensation='diff'
        )

        logger.info("Plotting multiband map...")
        plot_multiband(
            processed_map,
            bands=[(941, 10), (850, 10)],
            colors=[(1, 1, 0), (0, 0, 1)],
            method='median',
            compensation='raw'
        )

    def showCluster():

        logger.info("Computing k-means clustering...")
        labels = compute_kmeans(processed_map,
                                n_clusters=4,
                                method='median',
                                compensation='diff')
        plot_cluster(labels, figsize=(2500, 2500), cmap='Set1')

    def showPCA():

        logger.info("Computing PCA scores...")
        pca_scores = compute_pca(processed_map, n_components=3, method='diff')

        # Plota cada componente
        for i in range(pca_scores.shape[2]):
            plot_pca(pca_scores[:, :, i], component=i + 1, figsize=(2500, 2500), cmap='RdBu_r')


    file_path = "data/St kC CLs/Map St kC CL 14 Region 2.txt"

    logger.info("Loading data...")
    raw_map = load_file(file_path)

    logger.info("Preprocessing maps...")
    processed_map = preprocess([raw_map], region=(250, 1800), win_len=15)[0]

    # showBands()
    # showMultibands()
    # showCluster()
    showPCA()

    plt.show()