import os
from spectrus.io import load_spectrum
from spectrus.plot_utils import set_font
from spectrus.utils import combine_spectra
from spectrus.preprocessing import preprocess_batch
from spectrus.multivariate import compute_pca, plot_pca, plot_pca_scree, plot_pca_loadings, compute_kmeans, plot_clusters

def run_pca(data_folder="./data"):

    spectra_raw = []
    sample_info = []  # guarda info para combinar depois

    # 1️⃣ Carregar todos spectra brutos
    for group_folder in os.listdir(data_folder):
        group_path = os.path.join(data_folder, group_folder)
        if not os.path.isdir(group_path):
            continue

        all_files = [f for f in os.listdir(group_path) if "Map" not in f]

        concentrations = sorted(set([
            f.split("CL")[1].split("Region")[0].strip()
            for f in all_files if "CL" in f
        ]))

        for conc in concentrations:
            matching_files = [f for f in all_files if f"CL {conc}" in f]

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

        cluster_labels, kmeans_model = compute_kmeans(scores, n_clusters=3)
        plot_pca(scores, pca_model, labels=labels_final, title="PCA Score Plot + Clusters", kmeans_model=kmeans_model)


    else:
        print("⚠️ Dataset insuficiente para PCA (mínimo = 2 amostras).")

if __name__ == "__main__":
    font_path = (
        "C:/Users/petru/AppData/Local/Programs/Python/Python313/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/"
                 "helvetica-light-587ebe5a59211.ttf"
        )   
    set_font(font_path)

    run_pca()
