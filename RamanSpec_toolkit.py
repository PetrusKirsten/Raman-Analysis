# main.py

from spectrus.io import load_spectrum
from spectrus.plot_utils import set_font, plot_spectra, plot_stacked
from spectrus.preprocessing import preprocess
import matplotlib.pyplot as plt

from spectrus.utils import combine_spectra


# def rawVSprocessed():
#
#     from spectrus.io import load_spectrum
#     from spectrus.preprocessing import preprocess
#     from spectrus.plot_utils import set_font, plot_spectrum, compare_spectra
#
#     # 1. (opcional) Definir sua fonte personalizada
#     font_path = ("C:/Users/petru/AppData/Local/Programs/Python/Python313/"
#                  "Lib/site-packages/matplotlib/mpl-data/fonts/ttf/"
#                  "helvetica-light-587ebe5a59211.ttf")
#
#     set_font(font_path)
#
#     # 2. Carregar e pré-processar espectros de duas regiões
#     file1 = "data/St kC CLs/St kC CL 7 Region 1.txt"
#     file2 = "data/St kC CLs/St kC CL 7 Region 2.txt"
#
#     spec1 = preprocess_raw(load_spectrum(file1))
#     spec2 = preprocess(load_spectrum(file1))
#
#     # 3. Plotar espectro único (exemplo)
#     # plot_spectrum(spec1, title="Region 1 Processed Spectrum")
#
#     # 4. Plotar comparação das duas regiões
#     compare_spectra(
#         spectra=[spec1, spec2],
#         labels=["Raw", "Preprocessed"],
#         title="Comparison of Raw and Preprocessed"
#     )


def main():

    # 1. (opcional) Definir sua fonte personalizada
    font_path = ("C:/Users/petru/AppData/Local/Programs/Python/Python313/"
                 "Lib/site-packages/matplotlib/mpl-data/fonts/ttf/"
                 "helvetica-light-587ebe5a59211.ttf")

    set_font(font_path)

    # 2. Carregar e pré-processar espectros de duas regiões
    file1 = "data/St kC CLs/St kC CL 7 Region 1.txt"
    file2 = "data/St kC CLs/St kC CL 7 Region 2.txt"

    spec1 = preprocess(load_spectrum(file1))
    spec2 = preprocess(load_spectrum(file2))

    # 4. Plotar comparação das duas regiões
    plot_stacked(
        spectra=[spec1, spec2, combine_spectra([spec1, spec2])],
        labels=["Region 1", "Region 2", "Regions combined"],
        title="Region 1 and Region 2 spectra"
    )

    combine_spectra([spec1, spec2])


if __name__ == "__main__":
    main()
