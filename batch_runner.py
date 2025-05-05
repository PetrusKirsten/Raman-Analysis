"""
Batch Processing Script for RamanMap Toolkit
===========================================

This script automates batch execution of the RamanMap_toolkit functions:
- Loads and preprocesses multiple Raman map files
- Generates and saves topography, single-band, multiband, k-means, and PCA plots
- Uses progress bars and log messages for console feedback
"""

import logging
import coloredlogs
import numpy as np
from tqdm import tqdm
from pathlib import Path
import RamanMap_toolkit as rm
import matplotlib.pyplot as plt

from RamanMap_toolkit import sum_intensity


# --------------------------------------
# Batch Processing Function
# --------------------------------------
def batch_process(input_folder: str, output_folder: str):
    """
    Execute batch processing for all Raman map files in input_folder.

    :param input_folder: Directory containing .txt map files
    :type input_folder: str
    :param output_folder: Directory to save output figures
    :type output_folder: str
    """

    def log_config():
        # Logging Configuration

        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        coloredlogs.install(
            level='INFO',
            logger=logger,
            fmt='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
            level_styles={
                'info': {'color': 'green'},
                'warning': {'color': 'yellow'},
                'error': {'color': 'red', 'bold': True},
                'critical': {'color': 'red', 'bold': True, 'background': 'white'},
            },
            field_styles={
                'asctime': {'color': 'blue'},
                'levelname': {'color': 'white', 'bold': True},
            }
        )

        return logger

    def folders_config():
        input_path, output_path = Path(input_folder), Path(output_folder)
        svg_path = output_path / "svg"
        output_path.mkdir(exist_ok=True, parents=True); svg_path.mkdir(exist_ok=True, parents=True)

        return input_path, output_path, svg_path

    log = log_config()
    log.info(f"Initializing...")
    log.info(f"Checking paths and folders...")
    in_folder, out_folder, svg_folder = folders_config()

    # 1) Discover and load all map files
    raw_maps = []
    map_files = [f for f in in_folder.glob("*.txt") if "Map" in f.name]

    log.info(f"Found {len(map_files)} map files:")
    for f in tqdm(map_files, desc="Loading maps", unit="file"):
        raw_maps.append(rm.load_file(str(f)))

    # 2) Preprocess all maps at once
    proc_maps = []
    log.info("Preprocessing maps...")
    for m in raw_maps:

        proc_maps.append(rm.preprocess([m], region=REGION, win_len=WIN_LEN)[0])

    # 3) Define helper functions for each plot type

    def run_spectra(spec_base, spec_title):

        ax = rm.plot_mean_spectrum(
            proc,
            figsize=(10, 6),
            title=f"Mean Spectrum – {spec_title}",
            line_kwargs={'color': 'crimson', 'lw': 1.})

        png_path = out_folder / f"{spec_base}_spectrum.png"
        plt.gcf().savefig(png_path, dpi=300)

        svg_path = svg_folder / f"{spec_base}_spectrum.svg"
        plt.gcf().savefig(svg_path, dpi=300)

        plt.close()

    def run_outliers(proc_map, out_base, out_title, method):
        """Generate and save an outlier mask for any 2D map."""

        if hasattr(proc_map, 'spectral_data'):
            arr2d = sum_intensity(proc_map, method=method)
        else:
            arr2d = proc_map

        rm.plot_outlier_mask(arr2d, title=f"{out_title} - Outliers")

        png_path = out_folder / f"{out_base}_outliers.png"
        plt.gcf().savefig(png_path, dpi=300)

        svg_path = svg_folder / f"{out_base}_outliers.svg"
        plt.gcf().savefig(svg_path, dpi=300)
        plt.close()

    def run_topography(topo_proc, topo_base, topo_title):
        """Generate and save the total intensity (topography) map."""

        rm.plot_topography(topo_proc, title=f"{topo_title} - Topography")

        png_path = out_folder / f"{topo_base}_topography.png"
        plt.gcf().savefig(png_path, dpi=300)

        svg_path = svg_folder / f"{topo_base}_topography.svg"
        plt.gcf().savefig(svg_path, dpi=300)

        plt.close()

    def run_bands(bands_proc, bands_base, bands_title):
        """Generate and save individual band maps."""

        # compensated
        for center, width, label in BANDS:
            rm.plot_band(
                bands_proc,
                center=center,
                width=width,
                title=f"{bands_title} - {label}",
                cmap='inferno',
                method='median',
                compensation='diff')

            png_path = out_folder / f"{bands_base}_spectrum_diff.png"
            plt.gcf().savefig(png_path, dpi=300)

            svg_path = svg_folder / f"{bands_base}_spectrum_diff.svg"
            plt.gcf().savefig(svg_path, dpi=300)

            plt.close()

        # raw
        for center, width, label in BANDS:
            rm.plot_band(
                bands_proc,
                center=center,
                width=width,
                title=f"{bands_title} - {label}",
                cmap='inferno',
                method='median',
                compensation='raw'
            )

            png_path = out_folder / f"{bands_base}_spectrum_raw.png"
            plt.gcf().savefig(png_path, dpi=300)

            svg_path = svg_folder / f"{bands_base}_spectrum_raw.svg"
            plt.gcf().savefig(svg_path, dpi=300)

            plt.close()

    def run_multibands(multi_proc, multi_base, multibands_title):
        """Generate and save compensated and raw multiband RGB images."""

        if len(BANDS) >= 3:

            # compensated
            rm.plot_multiband(
                multi_proc,
                bands=[(c, w) for c, w, _ in BANDS[:3]],
                colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                method='median',
                compensation='diff'
            )

            png_path = out_folder / f"{multi_base}_spectrum_diff.png"
            plt.gcf().savefig(png_path, dpi=300)

            svg_path = svg_folder / f"{multi_base}_spectrum_diff.svg"
            plt.gcf().savefig(svg_path, dpi=300)

            plt.close()

            # raw
            rm.plot_multiband(
                multi_proc,
                bands=[(c, w) for c, w, _ in BANDS[:3]],
                colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                method='median',
                compensation='raw'
            )

            png_path = out_folder / f"{multi_base}_spectrum_raw.png"
            plt.gcf().savefig(png_path, dpi=300)

            svg_path = svg_folder / f"{multi_base}_spectrum_raw.svg"
            plt.gcf().savefig(svg_path, dpi=300)

            plt.close()

    def run_kmeans(kmeans_proc, kmeans_base, kmeans_title):
        """Generate and save k-means clustering map."""

        labels = rm.compute_kmeans(
            kmeans_proc,
            n_clusters=N_CLUSTERS,
            method='median',
            compensation='diff'
        )

        rm.plot_cluster(labels, figsize=(2500, 2500), cmap='tab20')
        plt.gcf().savefig(out_folder / f"{kmeans_base}_kmeans.png", dpi=300)
        plt.close()

    def run_pca(pca_proc, pca_base, pca_title):
        """Generate and save PCA component score maps."""

        scores = rm.compute_pca(
            pca_proc,
            n_components=PCA_COMPONENTS,
            method='diff'
        )

        for i in range(PCA_COMPONENTS):
            rm.plot_pca(
                scores[:, :, i],
                component=i + 1,
                figsize=(2500, 2500),
                cmap='RdBu_r'
            )
            plt.gcf().savefig(out_folder / f"{pca_base}_PCA{i + 1}.png", dpi=300)
            plt.close()

    def run_histogram(proc_map, hist_base, title):

        """Generate and save a pixel-value histogram."""
        rm.plot_histogram(
            proc_map,
            title=f"{title} - Histogram")

        png_path = out_folder / f"{hist_base}_spectrum_diff.png"
        plt.gcf().savefig(png_path, dpi=300)

        svg_path = svg_folder / f"{hist_base}_spectrum_diff.svg"
        plt.gcf().savefig(svg_path, dpi=300)

        plt.close()

    # 4) Iterate over each preprocessed map and generate selected plots
    for txt_file, proc in zip(map_files, proc_maps):

        # file name corrections
        raw_stem = txt_file.stem.removeprefix("Map ")
        base = raw_stem.replace(" ", "_")
        axis_title = raw_stem

        log.info(f"→ Generating figures for {txt_file.name}")

        log.info("Plotting spectra...")
        run_spectra(base, axis_title)

        log.info("Plotting outliers map...")
        run_outliers(proc, base, axis_title, method='mean')

        log.info("Plotting final map histogram...")
        run_histogram(proc, base, axis_title)

        if MAP_MODE == 'topography':
            log.info("Plotting topography map...")
            run_topography(proc, base, axis_title)

        elif MAP_MODE == 'bands':
            log.info("Plotting bands map...")
            run_bands(proc, base, axis_title)

        elif MAP_MODE == 'multi':
            log.info("Plotting multibands map...")
            run_multibands(proc, base, axis_title)

        elif MAP_MODE == 'k':
            log.info("Plotting k-means map...")
            run_kmeans(proc, base, axis_title)

        elif MAP_MODE == 'pca':
            log.info("Plotting PCA cluster map...")
            run_pca(proc, base, axis_title)

        else:
            print(f"MAP_MODE must be 'topography', 'bands', 'multi', 'k', or 'pca'.")

        log.info(f"{txt_file.name} figures done!")

    log.info(f"→ Done! All maps were saved in {out_folder.resolve()}")


if __name__ == "__main__":

    # Batch Parameters (adjust as needed)
    MAP_MODE       = 'topography'                    # choose the mode to the maps
    SAMPLES_NAME   = 'Carrageenans'                      # which folder/sample group iterate
    INPUT_FOLDER   = f"data/{SAMPLES_NAME}"          # folder containing .txt map files
    OUTPUT_FOLDER  = f"figures/maps/{SAMPLES_NAME}/{MAP_MODE}"  # where to save
    REGION         = (40, 1785)                      # spectral crop range (cm^-1)
    WIN_LEN        = 15                              # Savitzky-Golay window length
    N_CLUSTERS     = 4                               # number of clusters for k-means
    PCA_COMPONENTS = 3                               # number of PCA components to plot
    BANDS          = [                               # list of (center, width, label) for band maps
        (951, 10, "951 cm$^{-1}$"),
        (850, 10, "850 cm$^{-1}$"),
        (550, 20, "550 cm$^{-1}$"),
    ]

    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)