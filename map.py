import pandas as pd
import ramanspy as rp
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def ramanMicroscopy(
        fileTitle,
        filePath,
        regionToCrop,
        legend,
        lineColors,
        peakBands,
        plot_map,
        plot_spectra,
        save
):
    def configFigure():

        plt.style.use('seaborn-v0_8-ticks')
        plt.figure(figsize=(16, 5), facecolor='snow').canvas.manager.set_window_title(fileTitle + ' - Raman spectra')
        plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        plt.gca().xaxis.set_major_locator(MultipleLocator(100))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(25))

    def readData(directory_lists):

        maps = []
        for filename in directory_lists:

            try:
                data = pd.read_csv(filename)

                imageGrid = data['X-Axis']
                # delete grid column, transpose, convert to array and reshape to 100×100 by 1024 wavenumbers
                spectral_data = data.drop(columns='X-Axis').T.values.reshape(100, 100, 1024)
                mapping = rp.SpectralImage(spectral_data, imageGrid)

                maps.append(mapping)

            except Exception as e:
                print(f"Error processing the file {filename}: {e}")
                maps.append(None)

        return maps

    def preprocess(maps_lists, winLen):

        def pipeline(spec):

            routine = rp.preprocessing.Pipeline([
                rp.preprocessing.misc.Cropper(region=regionToCrop),
                rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
                rp.preprocessing.denoise.SavGol(window_length=winLen, polyorder=3),
                # rp.preprocessing.baseline.FABC(),
                # rp.preprocessing.normalise.MinMax(pixelwise=0, a=0, b=100),
                # rp.preprocessing.normalise.Vector(pixelwise=0),
            ])

            return routine.apply(spec)

        processed = []

        for ind_map in maps_lists:
            try:
                ind_map = pipeline(ind_map)
                processed.append(ind_map)

            except Exception as e:
                print(f"Error processin the map {ind_map}: {e}")
                processed.append(None)

        return processed

    def plotSpectra(img):
        configFigure()
        c = 0
        for region in range(len(img)):
            img[region][x-1, y-1].plot(label=f'Region {region + 1} at ({x}, {y})', color=f'C{c}', alpha=.75, lw=.85)
            # img[region].plot(peakBands, label=f'Region {region+1}', color=f'deeppink')
            c += 1

        plt.xlim(regionToCrop)
        # plt.subplots_adjust(
        #     wspace=.015, hspace=.060,
        #     top=.950, bottom=.080,
        #     left=.025, right=.850)
        plt.tight_layout()

        return img

    def plotMap(img):

        for region in range(len(img)):
            for band in peakBands:

                axImg = rp.plot.image(
                    img[region].band(band),
                    title=legend[region],
                    cbar_label=f"Peak intensity at {band} cm$^{-1}$",
                    color='indigo')

                if plot_spectra:
                    axImg.plot(x, y, 'ro', markersize=2, zorder=2)

    # def drawPeaks(bands):
    #
    #     for band in bands:
    #
    #         axSpec.axvline(
    #             band,
    #             label='test',
    #             color='whitesmoke',
    #             lw=10,
    #             ls='-',
    #             alpha=.9,
    #             zorder=-2)
    #
    #         axSpec.axvline(
    #             band,
    #             color='dimgray',
    #             lw=.75,
    #             ls=':',
    #             alpha=.8,
    #             zorder=-1)

    # def plotPeakDist(spectra, bands):
    #     for band in bands:
    #         fig = plt.figure(figsize=(8, 7), facecolor='snow')
    #         fig.canvas.manager.set_window_title(fileTitle + f' - peaks distribution at {band}')
    #         gs = GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
    #         axPeak = fig.add_subplot(gs[0, 0])
    #         axPeak.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
    #
    #         rp.plot.peak_dist(
    #             spectra, band,
    #             ax=axPeak,
    #             title=fileTitle + f' peaks distribution at {band} cm$^{{{-1}}}$',
    #             labels=legend,
    #             color=lineColors,
    #             alpha=.8,
    #             edgecolor='#383838',
    #             linewidth=.85,
    #             ecolor='#252525',
    #         )
    #
    #         plt.tight_layout()

    # create some vars
    # peaks_found, peaks_prop, axSpec = None, None, None

    # read & preprocess data
    raw_map = readData(filePath)
    processed_map = preprocess(raw_map, 16)

    x, y = 18, 97  # to plot the spectrum at this pixel

    if plot_spectra:
        plotSpectra(processed_map)

    if plot_map:
        plotMap(processed_map)

    if save:
        plt.savefig(f'{fileTitle}' + '.png', facecolor='snow', dpi=300)

    return processed_map


if __name__ == '__main__':
    ramanMicroscopy(
        'St CLs',
        [
            "data/St CLs/Map St CL 0 Region 1.txt",
            "data/St CLs/Map St CL 0 Region 2.txt",
            "data/St CLs/Map St CL 7 Region 1.txt",
            "data/St CLs/Map St CL 7 Region 2.txt",
        ],
        (100, 1800),  # all spectrum: (200, 1800); ideal: (300, 1500)
        [
            'St CL 0 Region I',
            'St CL 0 Region II',
            'St CL 7 Region I',
            'St CL 7 Region II'
        ],
        ['#E1C96B'],
        [478],  # starch principal peak: 478
        True,
        True,
        False)

    rp.plot.show()
