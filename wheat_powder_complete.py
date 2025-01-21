import numpy as np
import pandas as pd
import ramanspy as rs
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


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


filetitle = 'Wheat starch powder - Stacked'

configFigure()  # pyplot figure configs

# data importing and reading
filepath = [
    "data/WSt Powder 10x Region 1.txt",
    "data/WSt Powder 10x Region 2.txt",
    "data/WSt Powder 10x Region 3.txt",
]
powder_st = readData(filepath)

# pipelines

baseline = rs.preprocessing.Pipeline([
    rs.preprocessing.misc.Cropper(region=(35, 1800)),
    rs.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
    rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    rs.preprocessing.baseline.ASPLS(),
])
minmax = rs.preprocessing.Pipeline([
    rs.preprocessing.misc.Cropper(region=(35, 1800)),
    rs.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=25),
    rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    rs.preprocessing.baseline.ASPLS(),
    rs.preprocessing.normalise.MinMax(pixelwise=True),
])

# apply pipes
baseline_powder_st = baseline.apply(powder_st)
minmax_powder_st = minmax.apply(powder_st)

# plot and show data
raw_chart = rs.plot.spectra(
    minmax_powder_st[0],
    title=filetitle,
    plot_type='single',
    # label=['Cropping', 'Cosmic rays correction', 'Filtering', 'Baseline', 'MinMax'],
    # color=['crimson', 'mediumseagreen', 'royalblue'],
    color='orange',
    lw=1
)
_ = rs.plot.peaks(minmax_powder_st[0], prominence=0.01, color='orange')


# figure layout config
plt.xlim([0, 1800])
# plt.subplots_adjust(
#     wspace=0.015, hspace=0.060,
#     top=0.950, bottom=0.100,
#     left=0.075, right=0.840)
plt.tight_layout()

# plt.savefig(f'{filetitle}' + '.png', facecolor='w', dpi=300)
rs.plot.show()
