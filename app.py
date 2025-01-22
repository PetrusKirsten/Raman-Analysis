import numpy as np
import pandas as pd
import ramanspy as rs
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def spectrum(
        filetitle,
        filepath,
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

        for file in filenames:
            data = pd.read_csv(file)
            xData, yData = data['X-Axis'], data[data.keys()[-1]]
            spectrum_list.append(rs.Spectrum(yData, xData))

        return spectrum_list

    def preprocess(spec):
        pipeline = rs.preprocessing.Pipeline([
            rs.preprocessing.misc.Cropper(region=(35, 1800)),
            rs.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
            rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rs.preprocessing.baseline.ASPLS(),
            rs.preprocessing.normalise.MinMax(pixelwise=True),
        ])

        return pipeline.apply(spec)

    peaks_found, peaks_prop = None, None

    configFigure()
    raw_spectrum = readData(filepath)
    processed_spectrum = preprocess(raw_spectrum)

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
            label=['Region 1', 'Region 2', 'Region 3'],
            color=lineColor,
            lw=.85)

    if plot_peaks:
        _, peaks_found, peaks_prop = rs.plot.peaks(
            processed_spectrum[0],
            prominence=0.01,
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

    plt.xlim([0, 1800])
    # plt.subplots_adjust(
    #     wspace=0.015, hspace=0.060,
    #     top=0.950, bottom=0.100,
    #     left=0.075, right=0.840)
    plt.tight_layout()
    rs.plot.show()

    if save:
        plt.savefig(f'{filetitle}' + '.png', facecolor='w', dpi=300)

    if plot_peaks:
        return processed_spectrum, peaks_found, peaks_prop
    else:
        return processed_spectrum


if __name__ == '__main__':

    starch_powder = spectrum(
        'Wheat starch powder',
        [
            "data/WSt Powder 10x Region 1.txt",
            "data/WSt Powder 10x Region 2.txt",
            "data/WSt Powder 10x Region 3.txt",
        ],
        'darkorange',
        False, False, False,
    )



