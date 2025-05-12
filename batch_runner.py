"""
Batch Processing Script for RamanMap Toolkit
===========================================

This script automates batch execution of the RamanMap_toolkit functions:
- Loads and preprocesses multiple Raman map files
- Generates and saves topography, single-band, multiband, k-means, and PCA plots
- Uses progress bars and log messages for console feedback
"""

import logging
import time

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

def save_params(params: dict, output_folder: Path, filename: str):
    """
    Save run parameters to a text file.

    :param params: dict of {param_name: value}
    :param output_folder: Path where to write the file
    :param filename: name of the output file
    """

    fp = output_folder / filename

    with open(fp, 'w', encoding='utf-8') as f:

        f.write("RamanMap Batch Run Parameters\n")
        f.write("=============================\n\n")

        for k, v in params.items():
            f.write(f"{k:15s}: {v}\n")



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
                'info': {'color': 'white', 'bold': True},
                'warning': {'color': 'yellow'},
                'error': {'color': 'red', 'bold': True},
                'critical': {'color': 'red', 'bold': True, 'background': 'white'}},
            field_styles={
                'asctime': {'color': 'blue'},
                'levelname': {'color': 'white', 'bold': True}}
        )

        return logger

    def folders_config():
        input_path, output_path = Path(input_folder), Path(output_folder)
        svg_path = output_path / "svg"
        output_path.mkdir(exist_ok=True, parents=True); svg_path.mkdir(exist_ok=True, parents=True)

        return input_path, output_path, svg_path

    log = log_config()

    log.info(f"Initializing..."); time.sleep(1)
    log.info(f"Running for '{SAMPLES_NAME}'!"); time.sleep(1)
    log.info(f"Region to crop the spectra: {REGION[0]} to {REGION[1]} 1/cm."); time.sleep(.5)
    log.info(f"Creating {MAP_MODE} maps."); time.sleep(.5)
    log.info(f"Applying {IMAGE_FILTER} filter to the images."); time.sleep(.5)
    log.info(f"Checking paths and folders..."); time.sleep(.5)
    log.info(f"Input folder: '{INPUT_FOLDER}' OK! "); time.sleep(.5)
    log.info(f"Output folder: '{OUTPUT_FOLDER}' OK!"); time.sleep(.5)

    in_folder, out_folder, svg_folder = folders_config()

    # write and save the parameters of this exe
    run_params = {
        "MAP MODE": MAP_MODE,
        "REGION (in 1/cm)": REGION,
        "BACKGROUND REMOVED?": "NO" if REMOVE_BG == 'w-bg' else "YES",
        "WINDOW LENGTH FOR SAVGOL FILTER": WIN_LEN,
        "BANDS ANALYSED": BANDS,
    }
    log.info(f"Saving run parameters as '{SAMPLES_NAME}_run_params.txt'..."); time.sleep(.5)
    save_params(run_params, out_folder, f"{SAMPLES_NAME}_run_params.txt")

    # 1) Discover and load all map files
    raw_maps = []
    map_files = [f for f in in_folder.glob("*.txt") if "Map" in f.name]

    log.info(f"Found {len(map_files)} map files:"); time.sleep(.5)
    for f in map_files:
        print(f'\t\t→ {f.name}')
    time.sleep(.5)
    for f in tqdm(map_files, desc="Loading maps", unit="file"):
        raw_maps.append(rm.load_file(str(f)))

    # 2) Preprocess all maps at once
    proc_maps = []
    for m in tqdm(raw_maps, desc="Preprocessing maps", unit="maps"):
        proc_maps.append(rm.preprocess([m], region=REGION, win_len=WIN_LEN)[0])

    # Define helper functions for each plot type
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

        rm.plot_topography(
            topo_proc,
            title=f"{topo_title} - Topography", figsize=(2500, 2500),
            colormap='bone', im_filter=IMAGE_FILTER)

        png_path = out_folder / f"{topo_base}_topography.png"
        plt.gcf().savefig(png_path, dpi=300)

        svg_path = svg_folder / f"{topo_base}_topography.svg"
        plt.gcf().savefig(svg_path, dpi=300)

        plt.close()

    # def run_bands(bands_proc, bands_base, bands_title):
    #     """Generate and save individual band maps."""
    #
    #     # ratio compensation
    #     for center, width, label in BANDS:
    #         rm.plot_band(
    #             bands_proc,
    #             center=center,
    #             width=width,
    #             title=f"{bands_title} - {label}",
    #             cmap='PiYG',
    #             method='mean',
    #             im_filter=IMAGE_FILTER,
    #             compensation='ratio')
    #
    #         png_path = out_folder / f"{bands_base}_{center}_spectrum_ratio.png"
    #         plt.gcf().savefig(png_path, dpi=300)
    #
    #         svg_path = svg_folder / f"{bands_base}_{center}_spectrum_ratio.svg"
    #         plt.gcf().savefig(svg_path, dpi=300)
    #
    #         plt.close()
    #
    #     # subtraction compensation
    #     for center, width, label in BANDS:
    #         rm.plot_band(
    #             bands_proc,
    #             center=center,
    #             width=width,
    #             title=f"{bands_title} - {label}",
    #             cmap='plasma',
    #             method='mean',
    #             im_filter=IMAGE_FILTER,
    #             compensation='diff')
    #
    #         png_path = out_folder / f"{bands_base}_{center}_spectrum_sub.png"
    #         plt.gcf().savefig(png_path, dpi=300)
    #
    #         svg_path = svg_folder / f"{bands_base}_{center}_spectrum_sub.svg"
    #         plt.gcf().savefig(svg_path, dpi=300)
    #
    #         plt.close()
    #
    #     # raw
    #     for center, width, label in BANDS:
    #         rm.plot_band(
    #             bands_proc,
    #             center=center,
    #             width=width,
    #             title=f"{bands_title} - {label}",
    #             cmap='pink',
    #             method='mean',
    #             im_filter=IMAGE_FILTER,
    #             compensation='raw')
    #
    #         png_path = out_folder / f"{bands_base}_{center}_spectrum_raw.png"
    #         plt.gcf().savefig(png_path, dpi=300)
    #
    #         svg_path = svg_folder / f"{bands_base}_{center}_spectrum_raw.svg"
    #         plt.gcf().savefig(svg_path, dpi=300)
    #
    #         plt.close()

    def run_bands(proc, base, title, global_max):
        """Generate and save globally normalized band maps (raw + diff)."""

        for center, width, label in BANDS:

            with_st = 'St' in title
            with_kc = 'kC' in title
            with_ic = 'iC' in title

            if center == 480 and not with_st:
                continue

            if center == 805 and not with_ic:
                continue

            if center == 850 and not with_kc:
                continue

            if center == 941 and not with_st:
                continue

            if center == 1220 and not with_st:
                continue

            # RAW (global)
            rm.plot_band_global_norm(
                proc,
                title=f"{title} - {label}",
                center=center, width=width,
                global_max=global_max[center],
                compensation='raw',
                cmap='plasma'
            )

            filename = out_folder / f"{base}_band_{center}_raw_global"
            plt.gcf().savefig(filename.with_suffix('.png'), dpi=300)
            plt.gcf().savefig(svg_folder / filename.with_suffix('.svg').name, dpi=300)
            plt.close()

            # DIFF (global)
            rm.plot_band_global_norm(
                proc,
                title=f"{title} - {label}",
                center=center, width=width,
                global_max=global_max[center],
                compensation='diff',
                cmap='plasma'
            )

            filename = out_folder / f"{base}_band_{center}_diff_global"
            plt.gcf().savefig(filename.with_suffix('.png'), dpi=300)
            plt.gcf().savefig(svg_folder / filename.with_suffix('.svg').name, dpi=300)
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


    # 3a) Coletar mapas de banda para normalização global
    log.info("Collecting band values for global normalization...")
    band_values = {center: [] for center, _, _ in BANDS}

    for proc in proc_maps:

        for center, width, _ in BANDS:
            band_map = rm.extract_band(proc, center=center, width=width, method='mean')
            band_values[center].append(band_map)

    # Calcular máximo global por banda
    global_max = {center: np.max(np.stack(band_values[center])) for center in band_values}
    log.info("Global normalization maxima computed for each band.")

    # 4) Iterate over each preprocessed map and generate selected plots
    for txt_file, proc in zip(map_files, proc_maps):

        # file name corrections
        raw_stem = txt_file.stem.removeprefix("Map ")
        base = raw_stem.replace(" ", "_")
        axis_title = raw_stem

        log.info(f"→ Generating figures for {txt_file.name}:"); time.sleep(.5)

        log.info("Plotting spectra..."); time.sleep(0)
        run_spectra(base, axis_title)

        # log.info("Plotting outliers map..."); time.sleep(0)
        # run_outliers(proc, base, axis_title, method='mean')
        #
        # log.info("Plotting final map histogram..."); time.sleep(0)
        # run_histogram(proc, base, axis_title)

        if MAP_MODE == 'topography':
            log.info("Plotting topography map..."); time.sleep(0)
            run_topography(proc, base, axis_title)

        elif MAP_MODE == 'bands':
            log.info("Plotting band maps with global normalization...")
            run_bands(proc, base, axis_title, global_max)

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

        log.info(f"{txt_file.name} figures done!\n"); time.sleep(1)

    log.info(f"→ Done! All maps were saved in {out_folder.resolve()}")


if __name__ == "__main__":

    # Batch Parameters (adjust as needed)
    # for name in ['St iC CLs']:
    for name in ['St kC CLs', 'St iC CLs', 'Carrageenans']:
        for mode in ['topography', 'bands']:
            if mode == 'topography':
                continue
            for init_cut in [40]:
                for filtering in ['nearest']:

                    SAMPLES_NAME   = name  # which folder/sample group iterate
                    MAP_MODE       = mode  # choose the mode to the maps
                    REMOVE_BG      = 'w-bg'  # remove the background from spectra
                    IMAGE_FILTER   = filtering  # filter mode to apply into the final image
                    REGION         = (init_cut, 1785)  # spectral crop range (cm^-1)
                    WIN_LEN        = 15  # Savitzky-Golay window length
                    N_CLUSTERS     = 4  # number of clusters for k-means
                    PCA_COMPONENTS = 3  # number of PCA components to plot+
                    BANDS          = [  # list of (center, width, label) for band maps
                        (480, 10, "C–O–C 480 cm$^{-1}$ (starch)"),
                        (550, 20, "Ca$^{2+}$ 550 cm$^{-1}$ (cross-linking)"),
                        (805, 10, "iC 805 cm$^{-1}$"),
                        (850, 10, "kC 850 cm$^{-1}$"),
                        (941, 10, "C–O 941 cm$^{-1}$ (starch)"),
                        (1220, 10, "S=O 1220 cm$^{-1}$"),
                    ]

                    INPUT_FOLDER   = f"data/{SAMPLES_NAME}"  # folder containing .txt map files
                    OUTPUT_FOLDER  = f"figures/maps/{SAMPLES_NAME}/{MAP_MODE}_{REGION[0]}to{REGION[1]}_{REMOVE_BG}_{IMAGE_FILTER}"  # where to save

                    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)