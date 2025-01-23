import pandas as pd
import ramanspy as rs
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def raman(
        fileTitle,
        filePath,
        legend,
        lineColors,
        peakBands,
        plot_mean,
        plot_peaks,
        save
):

    def configFigure():

        plt.style.use('seaborn-v0_8-ticks')
        plt.figure(figsize=(16, 9), facecolor='snow').canvas.manager.set_window_title(fileTitle + ' - Raman spectra')
        plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        plt.gca().xaxis.set_major_locator(MultipleLocator(100))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(25))

    def readDatas(directory_lists):

        raw = []

        for directory_list in directory_lists:

            spectra = []
            for filename in directory_list:

                try:
                    data = pd.read_csv(filename)

                    xData, yData = data['X-Axis'], data[data.keys()[-1]]
                    raman_spectrum = rs.Spectrum(yData, xData)
                    spectra.append(raman_spectrum)

                except Exception as e:
                    print(f"Error processing the file {filename}: {e}")
                    spectra.append(None)

            raw.append(spectra)

        return raw

    def preprocess(spectra_lists):

        def pipeline(spec):

            routine = rs.preprocessing.Pipeline([
                rs.preprocessing.misc.Cropper(region=(200, 1800)),
                rs.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
                rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rs.preprocessing.baseline.ASPLS(),
                rs.preprocessing.normalise.MinMax(pixelwise=True),
            ])

            return routine.apply(spec)

        processed = []

        for spectra_list in spectra_lists:
            processed_sublist = []

            for spectrum in spectra_list:

                try:
                    processed_spectrum = pipeline(spectrum)
                    processed_sublist.append(processed_spectrum)

                except Exception as e:
                    print(f"Error processin the spectrum {spectrum}: {e}")
                    processed_sublist.append(None)

            processed.append(processed_sublist)

        return processed

    def drawPeaks(bands):

        for band in bands:

            axSpec.axvline(
                band,
                label='test',
                color='whitesmoke',
                lw=10,
                ls='-',
                alpha=.9,
                zorder=-2)

            axSpec.axvline(
                band,
                color='dimgray',
                lw=.75,
                ls=':',
                alpha=.8,
                zorder=-1)

    def plotPeakDist(spectra, bands):
        for band in bands:
            fig = plt.figure(figsize=(8, 7), facecolor='snow')
            fig.canvas.manager.set_window_title(fileTitle + f' - peaks distribution at {band}')
            gs = GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
            axPeak = fig.add_subplot(gs[0, 0])
            axPeak.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

            rs.plot.peak_dist(
                spectra, band,
                ax=axPeak,
                title=fileTitle + f' peaks distribution at {band} cm$^{{{-1}}}$',
                labels=legend,
                color=lineColors,
                alpha=.8,
                edgecolor='#383838',
                linewidth=.85,
                ecolor='#252525',
            )

            plt.tight_layout()

    # create some vars
    peaks_found, peaks_prop, axSpec = None, None, None
    if legend is None:
        legend = [f'Region {i + 1}' for i in range(len(filePath))]

    configFigure()

    # read & preprocess data
    raw_spectra = readDatas(filePath)
    processed_spectra = preprocess(raw_spectra)

    # TODO: try to auto the peak finder to directly pass to drawPeaks()

    if plot_mean:
        axSpec = rs.plot.mean_spectra(
            processed_spectra,
            title=fileTitle,
            plot_type='single stacked',
            dist=False,
            label=legend,
            color=lineColors,
            lw=1)

    else:
        axSpec = rs.plot.spectra(
            processed_spectra,
            title=fileTitle,
            plot_type='single stacked',
            label=legend,
            color=lineColors,
            lw=1)

    if plot_peaks:
        for spectrum in processed_spectra:
            _, peaks_found, peaks_prop = rs.plot.peaks(
                spectrum[0],
                prominence=0.15,
                color=lineColors,
                lw=.5,
                return_peaks=True)

    drawPeaks(peakBands)

    plt.xlim([185, 1800])
    plt.subplots_adjust(
        wspace=.015, hspace=.060,
        top=.950, bottom=.080,
        left=.025, right=.850)
    # plt.tight_layout()

    plotPeakDist(processed_spectra, peakBands)

    if save:
        plt.savefig(f'{fileTitle}' + '.png', facecolor='snow', dpi=300)

    if plot_peaks:
        return processed_spectra, peaks_found, peaks_prop

    else:
        return processed_spectra


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

    st_cl = raman(
        'St CLs',
        [
            [
                "data/Powders/WSt Powder 10x Region 1.txt",
                "data/Powders/WSt Powder 10x Region 2.txt",
                "data/Powders/WSt Powder 10x Region 3.txt"],
            ["data/St CL 0 Region 1.txt", "data/St CL 0 Region 2.txt"],
            ["data/St CL 7 Region 1.txt", "data/St CL 7 Region 2.txt"],
            ["data/St CL 14 Region 1.txt", "data/St CL 14 Region 2.txt"],
            ["data/St CL 21 Region 1.txt", "data/St CL 21 Region 2.txt"],
        ],
        ['St Powder', 'St CL 0', 'St CL 7', 'St CL 14', 'St CL 21'],
        ['dimgrey', '#E1C96B', '#FFE138', '#F1A836', '#E36E34'],
        [478, 1130],
        True, False, False)

    rs.plot.show()
