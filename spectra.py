import pandas as pd
import ramanspy as rp
# from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def ramanSpectrum(
        fileTitle,
        samples,
        filePath,
        regionToCrop,
        lineColors,
        peakBands,
        plot_mean,
        plot_peaks,
        find_peaks,
        save
):

    def configFigure():
        dpi = 300
        width, height = 1920*2 / dpi, 1080*2 / dpi

        plt.style.use('seaborn-v0_8-ticks')
        plt.figure(figsize=(width, height), facecolor='snow').canvas.manager.set_window_title(fileTitle + ' - Raman spectra')
        plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        plt.gca().xaxis.set_major_locator(MultipleLocator(100))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(25))

    def readData(directory_lists):

        raw = []

        for directory_list in directory_lists:

            spectra = []
            for filename in directory_list:

                try:
                    data = pd.read_csv(filename)

                    xData, yData = data['X-Axis'], data[data.keys()[-1]]
                    raman_spectrum = rp.Spectrum(yData, xData)

                    spectra.append(raman_spectrum)

                except Exception as e:
                    print(f"Error processing the file {filename}: {e}")
                    spectra.append(None)

            raw.append(spectra)

        return raw

    def preprocess(spectra_lists):

        def pipeline(spec):

            routine = rp.preprocessing.Pipeline([
                rp.preprocessing.misc.Cropper(region=regionToCrop),
                rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
                rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
                rp.preprocessing.baseline.ASLS(),
                # rp.preprocessing.normalise.Vector(pixelwise=True),
                # rp.preprocessing.normalise.AUC(pixelwise=True),
                rp.preprocessing.normalise.MinMax(pixelwise=False),
            ])

            return routine.apply(spec)

        processed, spectra = pipeline([item for sublist in spectra_lists.values() for item in sublist]), {}
        index = 0
        for key, sublist in spectra_lists.items():
            spectra[key] = processed[index: index + len(sublist)]
            index += len(sublist)

        return spectra

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

            rp.plot.peak_dist(
                spectra, band,
                ax=axPeak,
                title=fileTitle + f' peaks distribution at {band} cm$^{{{-1}}}$',
                labels=samples,
                color=lineColors,
                alpha=.8,
                edgecolor='#383838',
                linewidth=.85,
                ecolor='#252525',
            )

            plt.tight_layout()

    # create some vars
    peaks_found, peaks_prop, axSpec = None, None, None
    if samples is None:
        samples = [f'Region {i + 1}' for i in range(len(filePath))]

    configFigure()

    # read & preprocess data
    # TODO: try to auto the peak finder to directly pass to drawPeaks()
    raw_spectra = readData(filePath)
    processed_spectra = preprocess(dict(zip(samples, raw_spectra)))

    if plot_mean:
        axSpec = rp.plot.mean_spectra(
            list(processed_spectra.values()),
            title=fileTitle,
            plot_type='single stacked',
            dist=True,
            label=samples,
            color=lineColors,
            lw=1)

    else:
        axSpec = rp.plot.spectra(
            list(processed_spectra.values()),
            title=fileTitle,
            plot_type='single stacked',
            label=samples,
            color=lineColors,
            lw=1)

    if find_peaks:
        for spectrum in processed_spectra.values():
            _, peaks_found, peaks_prop = rp.plot.peaks(
                spectrum[0],
                prominence=0.15,
                color=lineColors,
                lw=.5,
                return_peaks=True)

    drawPeaks(peakBands)

    plt.xlim(regionToCrop)
    plt.subplots_adjust(
        wspace=.015, hspace=.060,
        top=.950, bottom=.080,
        left=.025, right=.880)
    # plt.tight_layout()

    if save:
        plt.savefig(f'{fileTitle}' + '.png', facecolor='snow', dpi=300)

    if plot_peaks:
        plotPeakDist(list(processed_spectra.values()), peakBands)

    return processed_spectra, peaks_found, peaks_prop if find_peaks else processed_spectra


if __name__ == '__main__':

    # precursors = ramanSpectrum(
    #     'Precursors in powder',
    #     ['St', 'kCar', 'CaCl2'],
    #     [
    #         [
    #             "data/Powders/WSt Powder 10x Region 1.txt",
    #             "data/Powders/WSt Powder 10x Region 2.txt",
    #             "data/Powders/WSt Powder 10x Region 3.txt"
    #         ],
    #         [
    #             "data/Powders/kCar Powder Region 1.txt",
    #             "data/Powders/kCar Powder Region 2.txt",
    #             "data/Powders/kCar Powder Region 3.txt"
    #         ],
    #         [
    #             "data/Powders/CaCl2 Powder Region 1.txt",
    #             "data/Powders/CaCl2 Powder Region 2.txt",
    #             "data/Powders/CaCl2 Powder Region 3.txt",
    #             "data/Powders/CaCl2 Powder Region 4.txt",
    #             "data/Powders/CaCl2 Powder Region 5.txt",
    #         ],
    #     ],
    #     (200, 1785),  # all spectrum: (200, 1800); ideal: (300, 1500)
    #     ['goldenrod', '#FF0831', 'lightsteelblue'],
    #     [],
    #     True, False, False, True
    # )

    # cacl = ramanSpectrum(
    #     'CaCl2 in powder - different regions',
    #     ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5'],
    #     [
    #             ["data/Powders/CaCl2 Powder Region 1.txt"],
    #             ["data/Powders/CaCl2 Powder Region 2.txt"],
    #             ["data/Powders/CaCl2 Powder Region 3.txt"],
    #             ["data/Powders/CaCl2 Powder Region 4.txt"],
    #             ["data/Powders/CaCl2 Powder Region 5.txt"],
    #     ],
    #     (200, 1785),  # all spectrum: (200, 1800); ideal: (300, 1500)
    #     ['C1', 'C2', 'C3', 'C4', 'C5'],
    #     [],
    #     False, False, False, True
    # )

    # st_cls = ramanSpectrum(
    #     'St CLs - normalized by 478 1/cm peak',
    #     ['St CL 0', 'St CL 7', 'St CL 14', 'St CL 21'],
    #     [
    #         [
    #             "data/St CLs/St CL 0 Region 1.txt",
    #             "data/St CLs/St CL 0 Region 2.txt"
    #         ],
    #         [
    #             "data/St CLs/St CL 7 Region 1.txt",
    #             "data/St CLs/St CL 7 Region 2.txt"
    #         ],
    #         [
    #             "data/St CLs/St CL 14 Region 1.txt",
    #             "data/St CLs/St CL 14 Region 2.txt"
    #         ],
    #         [
    #             "data/St CLs/St CL 21 Region 1.txt",
    #             "data/St CLs/St CL 21 Region 2.txt"
    #         ],
    #     ],
    #     (300, 1785),  # all spectrum: (200, 1800); ideal: (300, 1500)
    #     [
    #         # No CL
    #         '#E1C96B',
    #         # CL 7
    #         'gold',
    #         # CL 14
    #         '#F1A836',
    #         # CL 28
    #         '#E36E34',
    #     ],
    #     [478],
    #     True, True, False, False)

    st_kc_cls = ramanSpectrum(
        'St kCar CLs',
        ['St kCar CL 0', 'St kCar CL 7', 'St kCar CL 14', 'St kCar CL 21', ],
        [
            [
                "data/St kC CLs/St kC CL 0 Region 1.txt",
                "data/St kC CLs/St kC CL 0 Region 2.txt"
            ],
            [
                "data/St kC CLs/St kC CL 7 Region 1.txt",
                "data/St kC CLs/St kC CL 7 Region 2.txt"
            ],
            [
                "data/St kC CLs/St kC CL 14 Region 1.txt",
                "data/St kC CLs/St kC CL 14 Region 2.txt"
            ],
            [
                "data/St kC CLs/St kC CL 21 Region 1.txt",
                "data/St kC CLs/St kC CL 21 Region 2.txt"
            ],
        ],
        (200, 1785),  # all spectrum: (200, 1800); ideal: (300, 1500)
        ['lightpink', 'hotpink', 'deeppink', 'crimson'],
        [478],
        True, True, False, True)

    rp.plot.show()

