import pandas as pd
import ramanspy as rp
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

import pandas as pd
import ramanspy as rp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def ramanSpectrum(
        fileTitle, filePath, regionToCrop, legend,
        lineColors, peakBands, plot_mean, plot_peaks,
        find_peaks, save
):
    def configFigure():
        plt.style.use('seaborn-v0_8-ticks')
        plt.figure(figsize=(16, 9), facecolor='snow')
        plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        plt.gca().xaxis.set_major_locator(MultipleLocator(100))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(25))

    def readData(directory_lists):
        raw = [[rp.Spectrum(pd.read_csv(f)['X-Axis'], pd.read_csv(f).iloc[:, -1])
                if f else None for f in dir_list] for dir_list in directory_lists]
        return raw

    def preprocess(spectra_lists):
        pipeline = rp.preprocessing.Pipeline([
            rp.preprocessing.misc.Cropper(region=regionToCrop),
            rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
            rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
            rp.preprocessing.baseline.ASLS(),
            rp.preprocessing.normalise.MinMax(pixelwise=False),
        ])
        return [[pipeline.apply(spec) if spec else None for spec in spectra] for spectra in spectra_lists]

    def drawPeaks(bands):
        for band in bands:
            axSpec.axvline(band, color='dimgray', lw=.75, ls=':', alpha=.8, zorder=-1)

    def plotPeakDist(spectra, bands):
        for band in bands:
            fig, ax = plt.subplots(figsize=(8, 7), facecolor='snow')
            rp.plot.peak_dist(
                spectra, band, ax=ax,
                title=f'{fileTitle} peaks at {band} cm⁻¹',
                labels=legend, color=lineColors,
                alpha=.8, edgecolor='#383838', linewidth=.85, ecolor='#252525'
            )
            plt.tight_layout()

    configFigure()
    raw_spectra = readData(filePath)
    processed_spectra = preprocess(raw_spectra)

    plot_func = rp.plot.mean_spectra if plot_mean else rp.plot.spectra
    axSpec = plot_func(
        processed_spectra, title=fileTitle,
        plot_type='single stacked', label=legend or [f'Region {i + 1}' for i in range(len(filePath))],
        color=lineColors, lw=1
    )

    if find_peaks:
        peaks_found, peaks_prop = zip(
            *[rp.plot.peaks(spec[0], prominence=0.15, color=lineColors, lw=.5, return_peaks=True)
              if spec else (None, None) for spec in processed_spectra])
    else:
        peaks_found, peaks_prop = None, None

    drawPeaks(peakBands)
    plt.xlim(regionToCrop)
    plt.subplots_adjust(wspace=.015, hspace=.060, top=.950, bottom=.080, left=.025, right=.850)
    if save:
        plt.savefig(f'{fileTitle}.png', facecolor='snow', dpi=300)

    if plot_peaks:
        plotPeakDist(processed_spectra, peakBands)

    return (processed_spectra, peaks_found, peaks_prop) if find_peaks else processed_spectra


if __name__ == '__main__':

    # st_cls = ramanSpectrum(
    #     'St CLs',
    #     [
    #         ["data/Powders/WSt Powder 10x Region 1.txt",
    #          "data/Powders/WSt Powder 10x Region 2.txt",
    #          "data/Powders/WSt Powder 10x Region 3.txt"],
    #         ["data/St CLs/St CL 0 Region 1.txt", "data/St CLs/St CL 0 Region 2.txt"],
    #         ["data/St CLs/St CL 7 Region 1.txt", "data/St CLs/St CL 7 Region 2.txt"],
    #         ["data/St CLs/St CL 14 Region 1.txt", "data/St CLs/St CL 14 Region 2.txt"],
    #         ["data/St CLs/St CL 21 Region 1.txt", "data/St CLs/St CL 21 Region 2.txt"],
    #     ],  # TODO: add CaCl2 spectrum
    #     (300, 1500),  # all spectrum: (200, 1800); ideal: (300, 1500)
    #     ['St Powder', 'St CL 0', 'St CL 7', 'St CL 14', 'St CL 21'],
    #     ['dimgrey', '#E1C96B', '#FFE138', '#F1A836', '#E36E34'],
    #     [478, 578, 940],
    #     True, False, False, False)

    st_kc_cls = ramanSpectrum(
        'St kCar CLs',
        [
            [
                "data/Powders/WSt Powder 10x Region 1.txt",
                "data/Powders/WSt Powder 10x Region 2.txt",
                "data/Powders/WSt Powder 10x Region 3.txt"
            ],
            [
                "data/Powders/kCar Powder Region 1.txt",
                "data/Powders/kCar Powder Region 2.txt",
                "data/Powders/kCar Powder Region 3.txt"
            ],
            [
                "data/St kC CLs/St kC CL 0 Region 1.txt",
                "data/St kC CLs/St kC CL 0 Region 2.txt"
            ],
        ],
        (200, 1800),  # all spectrum: (200, 1800); ideal: (300, 1500)
        ['St Powder', 'kCar Powder', 'St kCar CL 0'],
        ['dimgrey', 'coral', 'crimson'],
        [941],
        True, True, False, False)

    rp.plot.show()

