import os

import numpy as np
import tikzplotlib
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM

from common.timeseries import TimeSeries
from common.utils import (generate_sine_noisy_peaks, plot_attributions,
                          train_test_split)
from nf_ti_adapter.lstm import LstmNfTiAdapter

os.environ["NIXTLA_ID_AS_COL"] = "1"

SIZE = 1000
HORIZON = 48
LEVELS = [68]
INPUT_SIZE = 2 * HORIZON
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

model = LSTM(
    input_size=INPUT_SIZE,
    inference_input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    max_steps=200,
    random_seed=40,
    early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
)

nf_ti_adapter = LstmNfTiAdapter(model, 1)

_time_series_noisy = generate_sine_noisy_peaks(size=SIZE)
_time_series_clean = generate_sine_noisy_peaks(size=SIZE, noise_mean=0, noise_std=0)
_time_series_sum = _time_series_noisy + _time_series_clean
_time_series_ds = np.arange(len(_time_series_noisy))

time_series_list = [
    TimeSeries(
        unique_id="sine_noisy_peaks",
        ds=_time_series_ds,
        y=_time_series_noisy,
    ),
    TimeSeries(
        unique_id="sine_clean",
        ds=_time_series_ds,
        y=_time_series_clean,
    ),
    TimeSeries(
        unique_id="sine_sum",
        ds=_time_series_ds,
        y=_time_series_sum,
    ),
]

# train test split
train_time_series_list, test_time_series_list = train_test_split(
    time_series_list, TRAIN_SPLIT
)

# plot time series list with train / test split borders as vertical lines
fig, axs = plt.subplots(len(time_series_list), 1, sharex="all")
for i, time_series in enumerate(time_series_list):
    axs[i].plot(time_series.y)
    axs[i].axvline(
        x=len(time_series.y) * TRAIN_SPLIT,
        color="r",
        label="train / test split",
        linestyle="--",
    )
    axs[i].set_ylim(-5, 5)
    axs[i].set_title(time_series.unique_id)
    fig.subplots_adjust(hspace=0.6)
plt.show()

nf_ti_adapter.fit_list(
    train_time_series_list, val_size=int(VAL_SPLIT * len(train_time_series_list[0].y))
)

# pick random INPUT_SIZE window from test set
# start_idx = random.randint(0, len(test_ds) - INPUT_SIZE)
start_idx = 98
test_input_list = [
    TimeSeries(
        unique_id=ts.unique_id,
        ds=ts.ds[start_idx : start_idx + INPUT_SIZE],
        y=ts.y[start_idx : start_idx + INPUT_SIZE],
    )
    for ts in test_time_series_list
]

predictions = nf_ti_adapter.predict_list_plot(test_input_list)

target_indices = list(range(len(predictions[f"{model}-loc"])))
attribution_list, negative_attribution_list = nf_ti_adapter.explain_multi(
    "TIG", target_indices, "-loc"
)
"""
plot_attributions(
    attribution_list,
    negative_attribution_list,
    "-loc",
    test_input_ds,
    test_input_y,
    predictions,
    model,
    "TIG",
)

target_indices = list(range(len(predictions[f"{model}-scale"])))
attribution_list, negative_attribution_list = nf_ti_adapter.explain(
    "TIG", target_indices, "-scale"
)

plot_attributions(
    attribution_list,
    negative_attribution_list,
    "-scale",
    test_input_ds,
    test_input_y,
    predictions,
    model,
    "TIG",
)
"""
