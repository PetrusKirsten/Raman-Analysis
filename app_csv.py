import numpy as np
import pandas as pd
import ramanspy as rs
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def readData(filenames):
    raman_spectrum_list = []

    for file in filenames:

        data = pd.read_csv(file)
        spectral_axis, spectral_data = data['X-Axis'], data[data.keys()[-1]]
        raman_spectrum_list.append(rs.Spectrum(spectral_data, spectral_axis))

    return raman_spectrum_list


# pyplot figure configs
plt.style.use('seaborn-v0_8-ticks')
plt.figure(figsize=(16, 6), facecolor='whitesmoke')
plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.gca().xaxis.set_minor_locator(MultipleLocator(25))

# data importing and reading
filepath = [
    "data/WSt Powder 10x Region 1.txt",
    "data/WSt Powder 10x Region 2.txt",
    "data/WSt Powder 10x Region 3.txt",
]

powder_st = readData(filepath)

# data preprocessing
cropping_pipeline = rs.preprocessing.Pipeline([
    rs.preprocessing.misc.Cropper(region=(35, 1800)),
])
cropped_powder_st = cropping_pipeline.apply(powder_st)

# plot and show data
raw_chart = rs.plot.spectra(
    cropped_powder_st,
    plot_type='single',
    title='Wheat starch powder',
    label=['Region 1', 'Region 2', 'Region 3'],
    color=['r', 'g', 'b'], lw=.75
)

# figure layout config
plt.xlim([0, 1800])
# plt.subplots_adjust(
#     wspace=0.015, hspace=0.060,
#     top=0.950, bottom=0.100,
#     left=0.060, right=0.880
# )
plt.tight_layout()

rs.plot.show()
