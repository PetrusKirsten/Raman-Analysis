import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from spectrus.io import load_spectrum
from spectrus.utils import combine_spectra
from spectrus.preprocessing import preprocess_batch
from spectrus.plot_utils import set_font, plot_stacked, config_figure
from spectrus.multivariate import compute_pca, plot_pca, plot_pca_scree, plot_pca_loadings
from spectrus.analysis import extract_band_areas, plot_band_by_formulation, plot_all_bands
from spectrus.analysis import extract_band_metrics, compute_ratio, plot_band_metric, plot_all_metrics


def run_pca(data_folder="./data"):

    spectra_raw = []
    sample_info = []  # guarda info para combinar depois

    # 1️⃣ Carregar todos spectra brutos
    for group_folder in os.listdir(data_folder):
        group_path = os.path.join(data_folder, group_folder)
        if not os.path.isdir(group_path) or not "St" in group_folder:
            continue

        all_files = [f for f in os.listdir(group_path) if "Map" not in f and f.endswith('.txt')]
        if not all_files:
            continue

        if "CL" in all_files[0]:
            concentrations = sorted(set([
                f.split("CL")[1].split("Region")[0].strip()
                for f in all_files if "CL" in f
            ]))
        
        else:
            concentrations = sorted(set([
                f.split("Region")[0].strip()
                for f in all_files
            ]))

        for conc in concentrations:
            matching_files = [f for f in all_files if f"{conc}" in f]

            for file in matching_files:
                file_path = os.path.join(group_path, file)
                spectrum = load_spectrum(file_path)
                spectra_raw.append(spectrum)

                sample_info.append({
                    "group": group_folder.replace(" CLs", ""),
                    "concentration": conc
                })

    # 2️⃣ Preprocessar todos de uma vez (batch)
    spectra_processed = preprocess_batch(spectra_raw)

    # 3️⃣ Combinar replicatas de mesma amostra
    sample_dict = {}
    for spectrum, info in zip(spectra_processed, sample_info):
        key = f"{info['group']} - {info['concentration']} mM"
        if key not in sample_dict:
            sample_dict[key] = []
        sample_dict[key].append(spectrum)

    spectra_final = []
    labels_final = []

    for key, reps in sample_dict.items():
        combined = combine_spectra(reps)
        spectra_final.append(combined)
        labels_final.append(key)

    # 4️⃣ Rodar PCA
    if len(spectra_final) >= 2:
        print(f"🔎 Dataset com {len(spectra_final)} amostras. Rodando PCA...")

        scores, loadings, pca_model = compute_pca(spectra_final, n_components=2)

        # Plots
        plot_pca(scores, pca_model, labels=labels_final, title="PCA Score Plot")

        plot_pca_scree(pca_model, title="PCA Variance Explained")
        
        spectral_axis = spectra_final[0].spectral_axis
        for n in range(len(loadings)):
            plot_pca_loadings(loadings, spectral_axis, pc=n+1)
            pass

        # cluster_labels, kmeans_model = compute_kmeans(scores, n_clusters=3)
        # plot_pca(scores, pca_model, labels=labels_final, title="PCA Score Plot + Clusters", kmeans_model=kmeans_model)

    else:
        print("⚠️ Dataset insuficiente para PCA (mínimo = 2 amostras).")


def run_spectra(data_folder="./data",
                save: bool = False,
                out_folder: str = "./figures/spectra"):
    """
    Carrega, preprocessa, combina replicatas e plota TODOS os espectros:
      • por grupo de polímero
      • por concentração de Ca²⁺
      • heatmap geral

    Parameters
    ----------
    data_folder : str
        Pasta raiz contendo subpastas "St CLs", "St kC CLs", "St iC CLs".
    save : bool
        Se True, salva cada figura em out_folder.
    out_folder : str
        Pasta onde salvar figuras (será criada se não existir).
    """
    os.makedirs(out_folder, exist_ok=True)

    # 1️⃣ Carregar todos espectros brutos
    spectra_raw = []
    sample_info = []
    for group in os.listdir(data_folder):
        group_path = os.path.join(data_folder, group)
        if not os.path.isdir(group_path):
            continue

        all_files = [f for f in os.listdir(group_path) if "Map" not in f]
        concentrations = sorted({ 
            f.split("CL")[1].split("Region")[0].strip()
            for f in all_files if "CL" in f
        })

        for conc in concentrations:
            for fname in all_files:
                if f"CL {conc}" in fname:
                    sp = load_spectrum(os.path.join(group_path, fname))
                    spectra_raw.append(sp)
                    sample_info.append({
                        "group": group.replace(" CLs", ""),
                        "conc": conc
                    })

    # 2️⃣ Preprocessar em batch
    spectra_proc = preprocess_batch(spectra_raw)

    # 3️⃣ Combinar replicatas de mesma amostra
    buckets = defaultdict(list)
    for sp, info in zip(spectra_proc, sample_info):
        key = (info["group"], info["conc"])
        buckets[key].append(sp)

    spectra_final, labels_final = [], []
    for (grp, conc), reps in buckets.items():
        avg = combine_spectra(reps)
        spectra_final.append(avg)
        labels_final.append((grp, conc))

    # 4️⃣ Plot por grupo de polímero
    by_group = defaultdict(list)
    for sp, (grp, conc) in zip(spectra_final, labels_final):
        by_group[grp].append((float(conc), sp))

    colors = {
        "St": ['#E1C96B', '#FFE138', '#F1A836', '#E36E34'],
        "St kC": ['hotpink', 'mediumvioletred', '#A251C3', '#773AD1'],
        "St iC": ['lightskyblue', '#62BDC1', '#31A887', '#08653A'],
    }
    for grp, lst in by_group.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        concs, specs = zip(*lst_sorted)
        labels = [f"{int(c)} mM" for c in concs]
        title = f"{grp}"

        plot_stacked(
            spectra=list(specs), labels=labels, title=title, colors=colors[grp],
            save=save, out_folder=out_folder, filename=f"spectra_{grp.replace(' ', '_')}.png"
        )
    # plt.show()
    
    # 5️⃣ Plot por concentração de Ca²⁺
    by_conc = defaultdict(list)
    for sp, (grp, conc) in zip(spectra_final, labels_final):
        by_conc[conc].append((grp, sp))

    for conc, lst in by_conc.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        grps, specs = zip(*lst_sorted)
        title = f"{int(float(conc))} mM CaCl$_2$"

        colors_conc = [colors[grps[0]][2], colors[grps[1]][1], colors[grps[2]][1]]
        plot_stacked(
            spectra=list(specs), labels=list(grps), title=title, colors=colors_conc,
            save=save, out_folder=out_folder, filename=f"spectra_{int(float(conc))}mM.png"
        )
    # plt.show()

    return spectra_final, labels_final


def run_spectra_precursors(
    data_folder="./data",
    save: bool = False,
    out_folder: str = "./figures/spectra"
    ):
    """
    Carrega, preprocessa, combina replicatas e plota TODOS os espectros:
      • por grupo de polímero
      • por concentração de Ca²⁺
      • heatmap geral

    Parameters
    ----------
    data_folder : str
        Pasta raiz contendo subpastas "St CLs", "St kC CLs", "St iC CLs".
    save : bool
        Se True, salva cada figura em out_folder.
    out_folder : str
        Pasta onde salvar figuras (será criada se não existir).
    """
    os.makedirs(out_folder, exist_ok=True)

    # 1️⃣ Carregar todos espectros brutos
    spectra_raw = []
    sample_info = []
    for group in os.listdir(data_folder):
        group_path = os.path.join(data_folder, group)

        if not os.path.isdir(group_path) or 'St' in group:
            continue

        all_files = [f for f in os.listdir(group_path) if "Map" not in f and f.endswith('.txt')]
        if not all_files:
            continue

        if "CL" in all_files[0]:
            concentrations = sorted(set([
                f.split("CL")[1].split("Region")[0].strip()
                for f in all_files if "CL" in f
            ]))
        
        else:
            concentrations = sorted(set([
                f.split("Region")[0].strip()
                for f in all_files
            ]))

        for conc in concentrations:
            for fname in all_files:
                if f"{conc}" in fname:
                    sp = load_spectrum(os.path.join(group_path, fname))
                    spectra_raw.append(sp)
                    sample_info.append({
                        "group": group,
                        "conc": conc
                    })

    # 2️⃣ Preprocessar em batch
    spectra_proc = preprocess_batch(spectra_raw)

    # 3️⃣ Combinar replicatas de mesma amostra
    buckets = defaultdict(list)
    for sp, info in zip(spectra_proc, sample_info):
        key = (info["group"], info["conc"])
        buckets[key].append(sp)

    spectra_final = []
    labels_final  = []
    for (grp, conc), reps in buckets.items():
        avg = combine_spectra(reps)
        spectra_final.append(avg)
        labels_final.append((grp, conc))

    # 4️⃣ Plot por grupo de polímero
    by_group = defaultdict(list)
    for sp, (grp, conc) in zip(spectra_final, labels_final):
        by_group[grp].append((conc, sp))

    colors = {
        "Carrageenans": ['lightskyblue', 'hotpink'],
        "Precursors": ['mediumslateblue', 'orange', 'deeppink'],
    }

    for grp, lst in by_group.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        concs, specs = zip(*lst_sorted)
        labels = [f"{c}" for c in concs]
        title = f"{grp} (5 wt.% hydrogel)" if "Carrageenans" in grp else f"{grp}"

        plot_stacked(
            spectra=list(specs), labels=labels, title=title, colors=colors[grp], peak_prominence=.2,
            save=save, out_folder=out_folder, filename=f"spectra_{grp.replace(' ', '_')}.png"
        )

    plt.show()

    return spectra_final, labels_final


def run_bands(spectra, labels):

    # 1) defina suas bandas
    bands = {
        "C–O–C 480": (470, 490),
        "Ring 942":  (932, 952),
        "OSO3 845":  (835, 855),
        "OSO3 1240": (1230, 1250),
        "C–O 1080":  (1070, 1090),
    }

    # 2) extraia áreas
    df_areas = extract_band_areas(
        spectra,      # lista de rp.Spectrum já combinados
        labels,       # lista de (group, conc)
        bands
    )

    # 3) plot banda a banda
    for band_name in bands:
        plot_band_by_formulation(
            df_areas, 
            band=band_name, 
            save=True, 
            out_folder="./figures/band_plots"
        )

    # 4) ou tudo de uma vez
    plot_all_bands(
        df_areas,
        bands=list(bands.keys()),
        save=True,
        out_folder="./figures/band_plots"
    )


def run_bands_metric(spectra, labels):

    bands = {  
        # TODO: justify each band
        "478":   (478 - 10,  478 + 10),
        "851":   (851 - 5,   851 + 5),
        "862":   (862 - 25,  862 + 15),
        "939":   (939 - 15,  939 + 15),
        "1080":  (1080 - 10, 1080 + 10),
        "1650":  (1650 - 40, 1650 + 40),
    }

    df_metrics = extract_band_metrics(spectra, labels, bands)
    df_metrics = compute_ratio(df_metrics, "851", "939")

    # plot_band_metric(df_metrics, "Area at 851 1/cm", "Area", out_folder="figures/bands", save=True)
    # plot_band_metric(df_metrics, "Area at 1650 1/cm", "Area", out_folder="figures/bands", save=True)
    plot_band_metric(df_metrics, "ratio_851_to_939", "Area", out_folder="figures/bands", save=True)


if __name__ == "__main__":
    font_path = (
        "C:/Users/petru/AppData/Local/Programs/"
        "Python/Python313/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/"
        "helvetica-light-587ebe5a59211.ttf"
    )   
    set_font(font_path)

    spec, lbls = run_spectra("./data", save=True, out_folder="./figures/spectra")
    # spec, lbls = run_spectra_precursors("./data", save=True, out_folder="./figures/spectra")
    run_bands_metric(spec, lbls)
    # run_pca()
