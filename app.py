import numpy as np
import ramanspy as rs
from matplotlib import pyplot as plt

# pyplot figure configs
plt.style.use('seaborn-v0_8-ticks')
plt.figure(figsize=(14, 8))
plt.gca().spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

# data importing and reading
filepath = [
    "data/WSt Powder 10x Region 1.mat",
    "data/WSt Powder 10x Region 2.mat",
    "data/WSt Powder 10x Region 3.mat",
]
powder_st = [rs.load.witec(file) for file in filepath]

# data preprocessing
preprocessing_pipeline = rs.preprocessing.Pipeline([
    rs.preprocessing.misc.Cropper(region=(6088, 8000)),
    # rs.preprocessing.despike.WhitakerHayes(),
    # rs.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
])
preprocessed_powder_st = preprocessing_pipeline.apply(powder_st)

# plot and show data
chart = rs.plot.spectra(
    preprocessed_powder_st,
    plot_type='single',
    title='Wheat starch powder',
    label=['Region 1', 'Region 2', 'Region 3'],
    color=['r', 'g', 'b'], lw=.75
)

# xticks values correction
xticks = range(6000, 8000, 100)
new_labels = [f"{int(tick - 6000)}" for tick in xticks]
plt.xticks(ticks=xticks, labels=new_labels)
plt.xlim([6085, 7840])
plt.subplots_adjust(
    wspace=0.015, hspace=0.060,
    top=0.960, bottom=0.070,
    left=0.060, right=0.880)

rs.plot.show()
