# Template do script analyze_bands.py
import pandas as pd
import numpy as np
from spectrus.analysis import extract_band_metrics, compute_ratio, plot_band_metric, plot_all_metrics
from spectrus.utils import load_combine_spectra

def main():
    # 1️⃣ Definir bandas
    bands = {
        "478":   (468, 488),
        "851":   (846, 856),
        "862":   (847, 877),
        "939":   (924, 954),
        "1080":  (1070, 1090),
        "1650":  (1610, 1690),
    }

    # 2️⃣ Carregar e combinar espectros
    spectra_final, labels_final = load_combine_spectra("./data")

    # 3️⃣ Extrair métricas
    df_metrics = extract_band_metrics(spectra_final, labels_final, bands)

    # 4️⃣ Calcular razões químicas
    df_metrics = compute_ratio(df_metrics, "851", "478")
    df_metrics = compute_ratio(df_metrics, "1650", "1080")

    # 5️⃣ Exploração univariada
    for band in bands:
        plot_band_metric(df_metrics, f"area_{band}", f"Área {band} cm⁻¹", out_folder="./figs/bands", save=True)
        plot_band_metric(df_metrics, f"center_{band}", f"Centro {band} cm⁻¹", out_folder="./figs/bands", save=True)
        plot_band_metric(df_metrics, f"fwhm_{band}", f"FWHM {band} cm⁻¹", out_folder="./figs/bands", save=True)

    # Razões
    plot_band_metric(df_metrics, "ratio_851_to_478", "Razão 851/478", out_folder="./figs/bands", save=True)
    plot_band_metric(df_metrics, "ratio_1650_to_1080", "Razão 1650/1080", out_folder="./figs/bands", save=True)

    # 6️⃣ Visão global
    plot_all_metrics(df_metrics, list(bands.keys()), out_folder="./figs/bands", save=True)

    # 7️⃣ Exportar resultados
    df_metrics.to_csv("./figs/bands/band_metrics.csv", index=False)
    print("Análise completa! Resultados salvos em ./figs/bands/")

if __name__ == "__main__":
    main()
