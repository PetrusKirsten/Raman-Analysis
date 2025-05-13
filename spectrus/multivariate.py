import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
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
            title="PCA Score Plot", size=(3500, 3000),
            save=False, save_path="pca_plot.png"):
    """
    PCA score plot with explained variance using toolkit style.
    """
    
    explained = pca_model.explained_variance_ratio_ * 100
    ax = config_figure(title, size)

    for i in range(scores.shape[0]):

        label = labels[i] if labels else f"S{i+1}"
        ax.scatter(scores[i, 0], scores[i, 1],
                   edgecolor='black', s=70, label=label,
                   color=f"C{i}", zorder=3)

        ax.annotate(label,
                    xy=(scores[i, 0], scores[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_pca_scree(pca_model, title="PCA Scree Plot", size=(4000, 1500)):
    """
    Plot Scree plot (variance explained by each PC).
    """
    
    explained = pca_model.explained_variance_ratio_ * 100

    n_components = len(explained)
    ax = config_figure(title, size)

    ax.bar(range(1, n_components + 1), explained,
           color='deepskyblue', edgecolor='black')
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_xticks(range(1, n_components + 1))
    ax.set_ylim(0, max(explained) * 1.2)

    plt.tight_layout()
    plt.show()

def plot_pca_loadings(loadings, spectral_axis,
                      pc=1, title="PCA Loading Plot", size=(4000, 1500)):
    """
    Plot loadings for a given principal component.

    Parameters
    ----------
    loadings : np.ndarray
        PCA loadings array (components x variables).
    spectral_axis : np.ndarray
        Raman Shift axis.
    pc : int
        Which PC to plot (1 = PC1, 2 = PC2, etc.).
    """
    
    ax = config_figure(title + f" PC{pc}", size)
    pc_index = pc - 1
    
    ax.plot(spectral_axis, loadings[pc_index],
            color="#383838", lw=1.)
    
    ax.axhline(0, lw=.75, ls=':', color='#383838', zorder=-1)

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Loading Value")
    ax.set_xlim((min(spectral_axis), max(spectral_axis)))
    
    plt.tight_layout()
    plt.show()

def plot_pca_loadings_with_peaks(loadings, spectral_axis,
                                 pc=1, n_peaks=5,
                                 min_distance=20, prominence=0.01,
                                 title="PCA Loading Plot with Peaks",
                                 size=(4000, 1500)):
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

    ax.plot(spectral_axis, loading, color="black", lw=1.5)
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Loading Value")

    peaks, _ = find_peaks(np.abs(loading), distance=min_distance, prominence=prominence)

    # pegar os N maiores
    peak_heights = np.abs(loading[peaks])
    top_indices = peaks[np.argsort(peak_heights)[-n_peaks:]]

    peak_positions = spectral_axis[top_indices]
    peak_values = loading[top_indices]

    for x, y in zip(peak_positions, peak_values):
        ax.axvline(x=x, color='red', linestyle=':', lw=1)
        ax.annotate(f"{int(x)}", xy=(x, y),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.show()