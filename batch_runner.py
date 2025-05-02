import logging
from tqdm import tqdm
from pathlib import Path
import RamanMap_toolkit as rm
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# --------------------------------------
# Parâmetros de batch (ajuste aqui)
# --------------------------------------
INPUT_FOLDER  = "data/St CLs"            # pasta com os .txt
OUTPUT_FOLDER = "figures/maps/St CLs/segmentos"                # onde salvar figuras
REGION        = (250, 1800)              # crop espectral
WIN_LEN       = 15                       # SavGol window
BANDS         = [                        # (center, width, nome)
    (951, 10, "951 cm$^{-1}$"),
    (850, 10, "850 cm$^{-1}$"),
    (550, 20, "550 cm$^{-1}"),]
N_CLUSTERS    = 4
PCA_COMPONENTS= 3

# --------------------------------------
# Função de processamento em lote
# --------------------------------------
def batch_process(
    input_folder: str,
    output_folder: str
):

    def showTopography():
        rm.plot_topography(proc, title=f"{axisTitle} - Topography")
        plt.gcf().savefig(output_path / f"{base}_topo.png", dpi=300)
        plt.gcf().savefig(output_path / f"{base}_topo.svg", dpi=300); plt.close(plt.gcf())

    def showBands():
        for center, width, name in BANDS:

            rm.plot_band(
                proc,
                center=center,
                width=width,
                title=f"{axisTitle} - {name}",
                cmap='inferno',
                method='median',
                compensation='raw')

            plt.gcf().savefig(output_path / f"{base}_band_{center}.png"); plt.close(plt.gcf())

    def showMultibands():
        if len(BANDS) >= 3:
            rm.plot_multiband(
                proc,
                bands         = [(c,w) for c,w,_ in BANDS[:3]],
                colors        = [(1,0,0),(0,1,0),(0,0,1)],
                method        = 'median',
                compensation  = 'raw')

            plt.gcf().savefig(output_path / f"{base}_multiband.png"); plt.close(plt.gcf())

    def showKmeans():
        labels = rm.compute_kmeans(
            proc,
            n_clusters   = N_CLUSTERS,
            method       = 'median',
            compensation = 'diff')

        rm.plot_cluster(labels, figsize=(2500,2500), cmap='tab20')
        plt.gcf().savefig(output_path / f"{base}_kmeans.png"); plt.close(plt.gcf())

    def showPCA():
        scores = rm.compute_pca(proc,
                                n_components=PCA_COMPONENTS,
                                method='diff')

        for i in range(PCA_COMPONENTS):
            rm.plot_pca(scores[:,:,i], component=i+1, figsize=(2500,2500), cmap='RdBu_r')
            plt.gcf().savefig(output_path / f"{base}_PCA{i+1}.png"); plt.close(plt.gcf())

    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1) listagem e pré-carregamento de todos os mapas
    map_files = [f for f in input_path.glob("*.txt") if "Map" in f.name]
    logger.info(f"Found {len(map_files)} map files. Loading...")
    raw_maps = []
    for f in tqdm(map_files, desc="Loading maps", unit="file"):
        raw_maps.append(rm.load_file(str(f)))

    # 2) pré-processar todos de uma vez
    logger.info("Preprocessing maps...")
    proc_maps = []
    for m in tqdm(raw_maps, desc="Preprocessing maps", unit="map"):
        proc_maps.append(rm.preprocess([m], region=REGION, win_len=WIN_LEN)[0])

    # 3) agora iteramos sobre cada arquivo já pré-processado
    for txt_file, proc in zip(tqdm(map_files, desc="Plotting maps"), proc_maps):

        base = txt_file.stem.removeprefix("Map ").replace(" ", "_")
        axisTitle = txt_file.stem.removeprefix("Map ")

        logger.info(f"→ Generating {txt_file.name}")

        # showTopography()
        # showBands()
        # showMultibands()
        showKmeans()
        # showPCA()

# --------------------------------------
# Entry point
# --------------------------------------
if __name__ == "__main__":
    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)
