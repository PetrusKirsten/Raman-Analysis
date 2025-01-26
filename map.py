import numpy as np
import pandas as pd
import ramanspy as rp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def ramanMicroscopy(
        fileTitle,
        filePath,
        regionToCrop,
        legend,
        peakBands,
        bandsColor,
        plot_map,
        plot_spectra,
        save
):
    def configFigure(size, face='snow'):

        fig = plt.figure(figsize=size, facecolor=face)
        # fig.canvas.manager.set_window_title(fileTitle + f' - peaks distribution at {band}')
        gs = GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
        ax = fig.add_subplot(gs[0, 0])
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        return ax

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

    def wnIndex(xAxis, wavenumber):  # find the nearest index to wavenumber
        return np.abs(xAxis - wavenumber).argmin()

    def plotSpectra(img):

        axSpec = configFigure((16, 5))

        c = 0
        for region in range(len(img)):
            img[region][x - 1, y - 1].plot(
                ax=axSpec, title='',
                label=f'Region {region + 1} at ({x}, {y})',
                color=f'C{c}', alpha=.75, lw=.85)
            c += 1

            img[region].mean.plot(
                ax=axSpec, title='',
                label=f'Region {region + 1} Mean',
                color=f'#383838', alpha=.75, lw=.85)

        drawPeaks(axSpec, peakBands)
        axSpec.set_xlim(regionToCrop)
        axSpec.tick_params(axis='y', which='both', left=False, labelleft=False)
        axSpec.xaxis.set_major_locator(MultipleLocator(100)), axSpec.xaxis.set_minor_locator(MultipleLocator(25))
        plt.tight_layout()

        return img

    def plotAnalysis(img):
        # TODO: read about unmixing & decomposition methods

        for region in range(len(img)):
            for band, color in zip(peakBands, bandsColor):

                pca = rp.analysis.decompose.PCA(n_components=4)
                projections, components = pca.apply(img[region])
                print(projections)

                plt.title(f'{legend[region]} | Peak intensity at {band} cm$^{{-1}}$')

                rp.plot.spectra(
                    components,
                    img[region].spectral_axis,
                    plot_type="single stacked",
                    color='red',
                    label=[f"Component {i + 1}" for i in range(len(components))])

                rp.plot.image(
                    projections,
                    color='red',
                    title=[f"Projection {i + 1}" for i in range(len(projections))])

                plt.tight_layout()

    def plotMicroscopy(img):

        def threshold(data):

            def condition(array):
                return array > array.mean()

            return np.where(condition(data), 1, 0).astype(float)

        for region in range(len(img)):
            for band, color in zip(peakBands, bandsColor):

                dataArray, wavenumbers = img[region].spectral_data, img[region].spectral_axis

                # TODO: choose appropriate way to quantify the map
                a, b = wnIndex(wavenumbers, regionToCrop[0]), wnIndex(wavenumbers, regionToCrop[-1])
                image_range = dataArray[:, :, wnIndex(wavenumbers, band)]
                image_range = np.sum(dataArray[:, :, a:b], axis=2)
                # # image_range = np.mean(dataArray[:, :, wnIndex-10:wnIndex+10], axis=2)
                # image_max = np.max(dataArray, axis=2)
                # image_mean = np.mean(dataArray, axis=2)
                alphaArray = threshold(image_range)

                axMap = configFigure((9, 7), face='w')
                plt.title(f'{legend[region]} | Peak intensity at {band} cm$^{{-1}}$')

                # TODO: try to merge some maps
                im = axMap.imshow(
                    image_range,
                    alpha=1.,
                    cmap=color,
                    interpolation='none',)

                cbar = plt.colorbar(im, ax=axMap, label='')
                cbar.set_ticks([])

                if plot_spectra:
                    axMap.plot(x, y, 'ro', markersize=2, zorder=2)

                axMap.tick_params(
                    axis='both', which='both',
                    left=False, labelleft=False,
                    bottom=False, labelbottom=False)
                plt.tight_layout()

    def drawPeaks(ax, bands):

        for band in bands:
            ax.axvline(
                band,
                label='test',
                color='whitesmoke',
                lw=10,
                ls='-',
                alpha=.9,
                zorder=-2)

            ax.axvline(
                band,
                color='dimgray',
                lw=.75,
                ls=':',
                alpha=.8,
                zorder=-1)

    # read & preprocess data
    plt.style.use('seaborn-v0_8-ticks')

    raw_map = readData(filePath)
    processed_map = preprocess(raw_map, 16)

    x, y = 50, 50  # to plot the spectrum at this pixel

    if plot_spectra:
        plotSpectra(processed_map)

    if plot_map:
        plotAnalysis(processed_map)

    plotMicroscopy(processed_map)

    if save:
        plt.savefig(f'{fileTitle}' + '.png', facecolor='snow', dpi=300)

    return processed_map


if __name__ == '__main__':
    ramanMicroscopy(
        'St CLs',
        [
            "data/St CLs/Map St CL 0 Region 1.txt",
            "data/St CLs/Map St CL 0 Region 2.txt",
            # "data/St CLs/Map St CL 7 Region 1.txt",
            # "data/St CLs/Map St CL 7 Region 2.txt",
            # "data/St CLs/Map St CL 14 Region 1.txt",
            # "data/St CLs/Map St CL 14 Region 2.txt",
            # "data/St CLs/Map St CL 21 Region 1.txt",
            # "data/St CLs/Map St CL 21 Region 2.txt",
        ],
        (300, 1500),  # all spectrum: (200, 1800); ideal: (300, 1500)
        [
            'St CL 0 Region I',
            'St CL 0 Region II',
            # 'St CL 7 Region I',
            # 'St CL 7 Region II',
            # 'St CL 14 Region I',
            # 'St CL 14 Region II',
            # 'St CL 21 Region I',
            # 'St CL 21 Region II',
        ],
        [478],  # in wavenumber / Raman shift. Starch principal peak: 478 1/cm
        ['summer'],
        False,
        True,
        False)

    plt.show()
