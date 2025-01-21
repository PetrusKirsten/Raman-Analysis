import numpy as np
import ramanspy as rs
from matplotlib import pyplot as plt

# pyplot figure configs
plt.style.use('seaborn-v0_8-ticks')
plt.figure(figsize=(16, 6), facecolor='whitesmoke')
plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

# data importing and reading
filepath = [
    # "data/WSt Powder 10x Region 1.mat",
    # "data/WSt Powder 10x Region 2.mat",
    "data/WSt Powder 10x Region 3.mat",
]
powder_st = [rs.load.witec(file) for file in filepath]
# powder_st = rs.load.witec(filepath[0])

# data preprocessing
cropping_pipeline = rs.preprocessing.Pipeline([
    rs.preprocessing.misc.Cropper(region=(6088, 8000)),
])
cropped_powder_st = cropping_pipeline.apply(powder_st)

preprocessing_pipeline = rs.preprocessing.Pipeline([
    rs.preprocessing.despike.WhitakerHayes(),
    # rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
])
preprocessed_powder_st = preprocessing_pipeline.apply(cropped_powder_st)

# plot and show data
raw_chart = rs.plot.spectra(
    cropped_powder_st,
    plot_type='single',
    title='Wheat starch powder',
    label=['Region 3 raw'],
    # label=['Region 1', 'Region 2', 'Region 3'],
    # color=['r', 'g', 'b'], lw=.75
    color=['r'], lw=.5
)

crc_chart = rs.plot.spectra(
    preprocessed_powder_st,
    plot_type='single',
    title='Wheat starch powder',
    label=['Region 3 corrected'],
    # label=['Region 1', 'Region 2', 'Region 3'],
    # color=['r', 'g', 'b'], lw=.75
    color=['b'], lw=.5
)

# xticks values correction
xticks = range(6000, 8000, 100)
new_labels = [f"{int(tick - 6000)}" for tick in xticks]
plt.xticks(ticks=xticks, labels=new_labels)
plt.xlim([6085, 7840])

# _ = rs.plot.peaks(preprocessed_powder_st, prominence=2.5)

# figure layout config
plt.subplots_adjust(
    wspace=0.015, hspace=0.060,
    top=0.950, bottom=0.100,
    left=0.060, right=0.880)
plt.tight_layout()

rs.plot.show()
