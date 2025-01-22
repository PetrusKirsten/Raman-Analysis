import numpy as np
import pandas as pd
import ramanspy as rs
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def spectrum(
        filetitle,
        filepath,
        legend,
        lineColor,
        plot_mean,
        plot_peaks,
        save
):

    def configFigure():
        plt.style.use('seaborn-v0_8-ticks')
        plt.figure(figsize=(16, 6), facecolor='whitesmoke')
        plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        plt.gca().xaxis.set_major_locator(MultipleLocator(100))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(25))

    def readData(filenames):
        spectrum_list = []

        for file in filenames:  # TODO: update function to read and store lists of spectra
            data = pd.read_csv(file)
            xData, yData = data['X-Axis'], data[data.keys()[-1]]
            spectrum_list.append(rs.Spectrum(yData, xData))

        return spectrum_list

    def readDatas(directory_lists):
        all_raman_spectra = []
        for directory_list in directory_lists:
            raman_spectra = []  # Lista para armazenar os espectros de uma sub-lista de diretórios
            for filename in directory_list:
                try:
                    # Lê os dados do arquivo CSV
                    data = pd.read_csv(filename)

                    # Obtém os dados espectrais
                    spectral_data = data['X-Axis']
                    spectral_axis = data[data.keys()[-1]]

                    # Cria o espectro Raman
                    raman_spectrum = rs.Spectrum(spectral_data, spectral_axis)

                    # Adiciona o espectro à lista
                    raman_spectra.append(raman_spectrum)
                except Exception as e:
                    print(f"Erro ao processar o arquivo {filename}: {e}")
                    raman_spectra.append(None)  # Adiciona None para manter o alinhamento

            all_raman_spectra.append(raman_spectra)  # Adiciona a lista processada ao resultado final

        return all_raman_spectra

    def preprocess(spec):
        pipeline = rs.preprocessing.Pipeline([
            rs.preprocessing.misc.Cropper(region=(200, 1800)),
            rs.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
            rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rs.preprocessing.baseline.ASPLS(),
            rs.preprocessing.normalise.MinMax(pixelwise=True),
        ])

        return pipeline.apply(spec)

    peaks_found, peaks_prop = None, None

    if legend is None:
        legend = [f'Region {i + 1}' for i in range(len(filepath))]

    configFigure()
    raw_spectrum = readDatas(filepath)
    processed_spectrum = preprocess(spec for spec in raw_spectrum)  # TODO: fix processing

    if plot_mean:
        rs.plot.mean_spectra(
            processed_spectrum,
            title=filetitle,
            color=lineColor,
            lw=.85)
    else:
        rs.plot.spectra(
            processed_spectrum,
            title=filetitle,
            plot_type='single stacked',
            label=legend,
            color=lineColor,
            lw=.85)

    if plot_peaks:
        _, peaks_found, peaks_prop = rs.plot.peaks(
            processed_spectrum[0],
            prominence=0.15,
            color=lineColor,
            lw=.5,
            return_peaks=True)
        print(
            peaks_found,
            '\n\n',
            peaks_prop)

    # rs.plot.peak_dist(
    #     [processed_spectrum],
    #     478,
    #     title=filetitle + ' peaks distribution',
    #     # labels=['Region 1', 'Region 2', 'Region 3'],
    #     # labels=['Region 1'],
    #     color=lineColor)

    plt.xlim([185, 1800])
    # plt.subplots_adjust(
    #     wspace=0.015, hspace=0.060,
    #     top=0.950, bottom=0.100,
    #     left=0.075, right=0.840)
    plt.tight_layout()
    # rs.plot.show()

    if save:
        plt.savefig(f'{filetitle}' + '.png', facecolor='w', dpi=300)

    if plot_peaks:
        return processed_spectrum, peaks_found, peaks_prop
    else:
        return processed_spectrum


if __name__ == '__main__':

    # starch_powder = spectrum(
    #     'Wheat starch powder',
    #     [
    #         "data/Powders/WSt Powder 10x Region 1.txt",
    #         "data/Powders/WSt Powder 10x Region 2.txt",
    #         "data/Powders/WSt Powder 10x Region 3.txt",
    #     ],
    #     None,
    #     'orange',
    #     True, True, False)
    #
    # kappa_powder = spectrum(
    #     'Kappa carrageenan powder',
    #     [
    #         "data/Powders/kCar Powder Region 1.txt",
    #         "data/Powders/kCar Powder Region 2.txt",
    #         "data/Powders/kCar Powder Region 3.txt",
    #         # "data/Powders/kCar Powder Region 4.txt",  # fotobleached
    #     ],
    #     None,
    #     'deeppink',
    #     False, False, False)
    # # kappa_powder_fb = spectrum(
    # #     'Kappa carrageenan fotobleached powder',
    # #     [
    # #         "data/Powders/kCar Powder Region 4.txt",
    # #     ],
    # #     'deeppink',
    # #     True, True, False)
    # cacl_powder = spectrum(
    #     'CaCl$_2$ powder',
    #     [
    #         "data/Powders/CaCl2 Powder Region 1.txt",
    #         "data/Powders/CaCl2 Powder Region 2.txt",
    #         "data/Powders/CaCl2 Powder Region 3.txt",
    #     ],
    #     None,
    #     'mediumseagreen',
    #     False, False, False)

    st_cl = spectrum(
        'St CLs',
        [
            ["data/Powders/WSt Powder 10x Region 1.txt"],
            ["data/St CL 0 Region 1.txt"],
            ["data/St CL 7 Region 1.txt"],
            ["data/St CL 14 Region 1.txt"],
            ["data/St CL 21 Region 1.txt"],
        ],
        ['St Powder', 'St CL 0', 'St CL 7', 'St CL 14', 'St CL 21'],
        'crimson',
        False, False, False)

    rs.plot.show()
