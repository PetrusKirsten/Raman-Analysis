import numpy as np
from matplotlib import cm
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from spectrus.plot_utils import config_figure


def compute_pca(spectra_list: list, n_components: int = 2):
    """
    Compute PCA on list of Spectra.

    Parameters
    ----------
    spectra_list : list of rp.Spectrum
        List of spectra.
    n_components : int
        Number of components.

    Returns
    -------
    scores : np.ndarray
        PCA scores.
    loadings : np.ndarray
        PCA loadings.
    pca_model : PCA
        Fitted PCA model.
    """
    
    data_matrix = np.array([spectrum.spectral_data for spectrum in spectra_list])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_matrix)
    loadings = pca.components_
    
    return scores, loadings, pca

def plot_pca(scores, pca_model, labels=None,
            title="PCA Score Plot", size=(3000, 2750),
            kmeans_model=None, show_hull=True, show_ellipse=True, ellipse_alpha=0.2, ellipse_conf=0.95,
            save=False, save_path="./figures/pca_plot.png"):
    """
    PCA score plot with explained variance using toolkit style.
    """
    
    explained = pca_model.explained_variance_ratio_ * 100
    ax = config_figure(title, size)

    for i in range(scores.shape[0]):
        label = labels[i] if labels else f"S{i+1}"
        if 'kC' in label:
            color = 'crimson'
        elif 'iC' in label:
            color = 'dodgerblue'
        else:
            color = 'gold'

        ax.scatter(scores[i, 0], scores[i, 1],
                   color=color, edgecolor='black', 
                   s=135, linewidths=.75, alpha=.75,
                   label=label,
                   zorder=3)

        ax.annotate(label,
                    xy=(scores[i, 0], scores[i, 1]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=12)

    max_x, max_y = np.max(np.abs(scores[:, 0])) * 1.3, np.max(np.abs(scores[:, 1])) * 1.3

    ax.axhline(0, color='gray', alpha=.5, lw=.8, ls='-', zorder=-1)
    ax.axvline(0, color='gray', alpha=.5, lw=.8, ls='-', zorder=-1)
    
    if kmeans_model is not None:
        centers = kmeans_model.cluster_centers_
        cluster_labels = kmeans_model.labels_
        n_clusters = len(np.unique(cluster_labels))
        cmap = cm.get_cmap("tab10", n_clusters)

        ax.scatter(
            centers[:, 0], centers[:, 1],
            marker='X', s=200, color='black',
            label='Centroids', zorder=4
        )

        if show_hull:
            for cluster in range(n_clusters):
                pts = scores[cluster_labels == cluster]

                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    ax.fill(hull_pts[:,0], hull_pts[:,1],
                            color=cmap(cluster), alpha=0.2,
                            label=f"Hull {cluster+1}" if cluster==0 else None)
        
        # elipse de confiança
        if show_ellipse:
            mu = pts.mean(axis=0)
            cov = np.cov(pts, rowvar=False)
            # escala para o nível de confiança desejado
            r2 = chi2.ppf(ellipse_conf, df=2)
            vals, vecs = np.linalg.eigh(cov * r2)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:,order]
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            width, height = 2 * np.sqrt(vals)
            ell = Ellipse(xy=mu, width=width, height=height,
                            angle=theta, edgecolor=color,
                            facecolor=color, alpha=ellipse_alpha)
            ax.add_patch(ell)

    ax.set_xlim(-max_x, max_x); ax.set_ylim(-max_y, max_y)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(save_path, dpi=300)

def plot_pca_scree(pca_model, title="PCA Scree Plot", size=(1500, 1500)):
    """
    Plot Scree plot (variance explained by each PC).
    """
    
    explained = pca_model.explained_variance_ratio_ * 100

    n_components = len(explained)
    ax = config_figure(title, size)

    ax.bar(range(1, n_components + 1), explained,
           color='deepskyblue', edgecolor='black', width=.5)
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_xticks(range(1, n_components + 1))
    ax.set_ylim(0, max(explained) * 1.2)

    plt.tight_layout()
    plt.show()

def plot_pca_loadings(loadings, spectral_axis,
                      title="PCA Loading Plot with Peaks", size=(4000, 1500),
                      pc=1, n_peaks=10, min_distance=5, prominence=0.01,
                      save=False, save_path="./figures/pca_loadings"):
    """
    Plot PCA loadings + highlight main peaks using real peak detection.

    Parameters
    ----------
    loadings : np.ndarray
        PCA loadings array (components x variables).
    spectral_axis : np.ndarray
        Raman Shift axis.
    pc : int
        Which PC to plot (1 = PC1, 2 = PC2, etc.).
    n_peaks : int
        Number of top peaks to highlight.
    min_distance : int
        Minimum distance between peaks (in data points).
    prominence : float
        Minimum prominence of peaks to be considered.
    """
    
    ax = config_figure(title + f" PC{pc}", size)
    pc_index = pc - 1
    loading = loadings[pc_index]

    ax.plot(spectral_axis, loading, 
            color='slategray', lw=1., alpha=.85,
            zorder=3)

    peaks, _ = find_peaks(np.abs(loading), distance=min_distance, prominence=prominence)

    # pegar os N maiores
    peak_heights = np.abs(loading[peaks])
    top_indices = peaks[np.argsort(peak_heights)[-n_peaks:]]

    peak_positions = spectral_axis[top_indices]
    peak_values = loading[top_indices]

    for x, y in zip(peak_positions, peak_values):
        ax.axvline(x=x, color='palevioletred', linestyle='--', lw=.75, zorder=1)

        ax.annotate(f"{int(x)}", xy=(x, y),
                    xytext=(0, 10 if y > 0 else -15), textcoords='offset points',
                    ha='center', fontsize=10, color='crimson',
                    bbox=dict(
                        boxstyle='round,pad=0.15',
                        facecolor='white',
                        edgecolor='none'),
                    zorder=4)

    ax.axhline(0, color='gray', alpha=.5, lw=.75, ls='-', zorder=0)

    ax.set_xlabel("Raman Shift (cm^{-1})")
    ax.set_ylabel("Loading Value")
    ax.set_xlim((min(spectral_axis), max(spectral_axis)))
    ax.set_ylim((min(1.5*loading), max(1.5*loading)))

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(save_path + f'{pc}.png', dpi=300)


def compute_kmeans(scores: np.ndarray,
                   n_clusters: int = 3,
                   random_state: int = 0) -> tuple:
    """
    Apply K-means clustering on PCA scores.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores matrix (samples x components).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    cluster_labels : np.ndarray
        Array of cluster assignments (0 .. n_clusters-1).
    kmeans_model : KMeans
        Fitted KMeans model.
    """
    
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state)
    
    cluster_labels = kmeans.fit_predict(scores)
    
    return cluster_labels, kmeans

def plot_clusters(scores: np.ndarray,
                  cluster_labels: np.ndarray,
                  kmeans_model=None,
                  title: str = "K-means Clustering",
                  size: tuple = (3000, 2750),
                  save: bool = False,
                  save_path: str = "clusters_plot.png"):
    """
    Plot PCA scores colored by cluster assignment, with optional centroids.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores matrix (samples x components).
    cluster_labels : np.ndarray
        Cluster labels from compute_kmeans().
    kmeans_model : KMeans, optional
        If provided, will plot centroids.
    title : str
        Plot title.
    size : tuple
        Figure size in pixels.
    save : bool
        Whether to save figure.
    save_path : str
        Path for saving if save=True.
    """
    
    ax = config_figure(title, size)
    n_clusters = len(np.unique(cluster_labels))
    
    # choose a colormap
    cmap = cm.get_cmap("tab10", n_clusters)

    for cluster in range(n_clusters):
        idx = cluster_labels == cluster
        ax.scatter(
            scores[idx, 0], scores[idx, 1],
            s=100, color=cmap(cluster), edgecolor='black',
            label=f"Cluster {cluster+1}", alpha=0.8, zorder=3
        )

    # plot centroids if available
    if kmeans_model is not None:
        centers = kmeans_model.cluster_centers_
        ax.scatter(
            centers[:, 0], centers[:, 1],
            marker='X', s=200, color='black',
            label="Centroids", zorder=4
        )

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel(ax.get_xlabel())  # mantém label de PC1
    ax.set_ylabel(ax.get_ylabel())  # mantém label de PC2
    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(save_path, dpi=300)
