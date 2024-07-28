import os
import random

import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM
from scipy.signal import find_peaks
from synthetictime.simple_time_series import SimpleTimeSeries

from common.utils import plot_attributions
from nf_ti_adapter.lstm import LstmNfTiAdapter

os.environ["NIXTLA_ID_AS_COL"] = "1"

HORIZON = 48
LEVELS = [80, 90]
INPUT_SIZE = 2 * HORIZON
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.3

model = LSTM(
    input_size=INPUT_SIZE,
    inference_input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=2800,
    random_seed=40,
    # early_stop_patience_steps=5,
    val_check_steps=10,
    logger=TensorBoardLogger("logs"),
)

nf_ti_adapter = LstmNfTiAdapter(model, 1)
time_series = SimpleTimeSeries(
    size=8000,
    base_amplitude=2.0,
    base_frequency=0.05,
    base_noise_scale=0,
    base_noise_amplitude=0.5,
).synthetic_time_series
time_series_ds = np.arange(len(time_series))

# train test split
last_train_idx = int(len(time_series) * TRAIN_SPLIT)
train_y = time_series[:last_train_idx]
train_ds = time_series_ds[:last_train_idx]

test_y = time_series[last_train_idx:]
test_ds = time_series_ds[last_train_idx:]

peaks, _ = find_peaks(train_y)
valleys, _ = find_peaks(-train_y)

marked_valley = 0
for i, peak in enumerate(peaks):
    # skip peak with a probability
    if random.random() < 0.8 or i < marked_valley:
        continue
    sub_arr = train_y[peak - 3 : peak + 4]
    sub_arr += np.random.normal(0, 2, len(sub_arr))

    valley_idx = min(i + 2, len(valleys) - 1)
    marked_valley = valley_idx
    valley = valleys[valley_idx]
    sub_arr = train_y[valley - 3 : valley + 3]
    sub_arr += np.random.normal(0, 2, len(sub_arr))

# plot time series with train / test split borders as vertical lines
plt.plot(time_series)
# plt.axvline(x=last_train_idx, color="r", label="train / test split", linestyle="--")
plt.legend()
plt.title("Time Series")
plt.show()

nf_ti_adapter.fit(train_ds, train_y, val_size=int(VAL_SPLIT * len(train_y)))

test_input_ds = test_ds[:INPUT_SIZE]
test_input_y = test_y[:INPUT_SIZE]

test_peaks, _ = find_peaks(test_input_y)
peak = test_peaks[-1]
sub_arr = test_input_y[peak - 3 : peak + 3]
sub_arr += np.random.normal(0, 2, len(sub_arr))

predictions = nf_ti_adapter.predict_plot(ds=test_input_ds, y=test_input_y)

target_indices = list(range(len(predictions[f"{model}-loc"])))
attribution_list, negative_attribution_list = nf_ti_adapter.explain(
    "TIG", target_indices, "-loc", test_input_ds, test_input_y
)

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
    "TIG", target_indices, "-scale", test_input_ds, test_input_y
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
