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
from tqdm import tqdm
from pathlib import Path
import RamanMap_toolkit as rm
import matplotlib.pyplot as plt


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

        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)

        coloredlogs.install(
            level='INFO',
            logger=log,
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
        return log

    logger = log_config()

    input_path, output_path = Path(input_folder), Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1) Discover and load all map files
    raw_maps = []
    map_files = [f for f in input_path.glob("*.txt") if "Map" in f.name]
    logger.info(f"Found {len(map_files)} map files. Loading...")
    for f in tqdm(map_files, desc="Loading maps", unit="file"):
        raw_maps.append(rm.load_file(str(f)))

    # 2) Preprocess all maps at once
    proc_maps = []
    logger.info("Preprocessing maps...")
    for m in tqdm(raw_maps, desc="Preprocessing maps", unit="map"):
        proc_maps.append(rm.preprocess([m], region=REGION, win_len=WIN_LEN)[0])

    # 3) Define helper functions for each plot type
        def show_topography(topo_proc, topo_base, topo_title):
            """Generate and save the total intensity (topography) map."""

            rm.plot_topography(topo_proc, title=f"{topo_title} - Topography")
            plt.gcf().savefig(output_path / f"{topo_base}_topo.png", dpi=300)
            plt.gcf().savefig(output_path / f"{topo_base}_topo.svg", dpi=300)
            plt.close()

        def show_bands(bands_proc, bands_base, bands_title):
            """Generate and save individual band maps."""

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

                plt.gcf().savefig(output_path / f"{bands_base}_band_{center}.png", dpi=300)
                plt.close()

        def show_multibands(multi_proc, multi_base, multibands_title):
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
                plt.gcf().savefig(output_path / f"{multi_base}_multiband_diff.png", dpi=300)
                plt.close()

                # raw
                rm.plot_multiband(
                    multi_proc,
                    bands=[(c, w) for c, w, _ in BANDS[:3]],
                    colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                    method='median',
                    compensation='raw'
                )
                plt.gcf().savefig(output_path / f"{multi_base}_multiband_raw.png", dpi=300)
                plt.close()

        def show_kmeans(kmeans_proc, kmeans_base, kmeans_title):
            """Generate and save k-means clustering map."""

            labels = rm.compute_kmeans(
                kmeans_proc,
                n_clusters=N_CLUSTERS,
                method='median',
                compensation='diff'
            )

            rm.plot_cluster(labels, figsize=(2500, 2500), cmap='tab20')
            plt.gcf().savefig(output_path / f"{kmeans_base}_kmeans.png", dpi=300)
            plt.close()

        def show_pca(pca_proc, pca_base, pca_title):
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
                plt.gcf().savefig(output_path / f"{pca_base}_PCA{i + 1}.png", dpi=300)
                plt.close()

    # 4) Iterate over each preprocessed map and generate selected plots
        for txt_file, proc in zip(tqdm(map_files, desc="Plotting maps", unit="map"), proc_maps):

            raw_stem = txt_file.stem.removeprefix("Map ")
            base = raw_stem.replace(" ", "_")
            axis_title = raw_stem

            logger.info(f"→ Generating figures for {txt_file.name}")

            if MAP_MODE == 'topography':
                show_topography(proc, base, axis_title)

            elif MAP_MODE == 'bands':
                show_bands(proc, base, axis_title)

            elif MAP_MODE == 'multi':
                show_multibands(proc, base, axis_title)

            elif MAP_MODE == 'k':
                show_kmeans(proc, base, axis_title)

            elif MAP_MODE == 'pca':
                show_pca(proc, base, axis_title)

            else:
                print(f"MAP_MODE must be 'topography', 'bands', 'multi', 'k', or 'pca'.")

    logger.info("\n\nDone! Maps generated!")


if __name__ == "__main__":

    # Batch Parameters (adjust as needed)
    MAP_MODE       = 'topography'                    # choose the mode to the maps
    SAMPLES_NAME   = 'St CLs'                        # which folder/sample group iterate
    INPUT_FOLDER   = f"data/{SAMPLES_NAME}"          # folder containing .txt map files
    OUTPUT_FOLDER  = f"figures/maps/{SAMPLES_NAME}/{MAP_MODE}"  # where to save
    REGION         = (250, 1800)                     # spectral crop range (cm^-1)
    WIN_LEN        = 15                              # Savitzky-Golay window length
    N_CLUSTERS     = 4                               # number of clusters for k-means
    PCA_COMPONENTS = 3                               # number of PCA components to plot
    BANDS          = [                               # list of (center, width, label) for band maps
        (951, 10, "951 cm$^{-1}$"),
        (850, 10, "850 cm$^{-1}$"),
        (550, 20, "550 cm$^{-1}$"),
    ]

    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)