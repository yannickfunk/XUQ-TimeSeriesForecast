import os
import random

import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM, NHITS
from scipy.signal import find_peaks
from synthetictime.simple_time_series import SimpleTimeSeries

from nf_ti_adapter.lstm import LstmNfTiAdapter
from nf_ti_adapter.nhits import NhitsNfTiAdapter

os.environ["NIXTLA_ID_AS_COL"] = "1"

HORIZON = 24
LEVELS = [80, 90]
INPUT_SIZE = 2 * HORIZON
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

model = NHITS(
    input_size=INPUT_SIZE,
    # inference_input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    # max_steps=1399,
    random_seed=42,
    early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
)

nf_ti_adapter = NhitsNfTiAdapter(model, 1)
time_series = SimpleTimeSeries(
    size=1000,
    base_amplitude=2.0,
    base_frequency=0.03,
    base_noise_scale=0,
    base_noise_amplitude=0.5,
).synthetic_time_series
time_series_ds = np.arange(len(time_series))

peaks, _ = find_peaks(time_series)
for peak in peaks:
    # skip peak with a probability of 0.5
    if random.random() < 0.5:
        continue
    sub_arr = time_series[peak - 5 : peak + 5]
    sub_arr += np.random.normal(0, 3, len(sub_arr))

negative_peaks, _ = find_peaks(-time_series)
for peak in negative_peaks:
    # skip peak with a probability of 0.8
    if random.random() < 0.5:
        continue
    sub_arr = time_series[peak - 5 : peak + 5]
    sub_arr += np.random.normal(0, 3, len(sub_arr))

# train test split
last_train_idx = int(len(time_series) * TRAIN_SPLIT)
train_y = time_series[:last_train_idx]
train_ds = time_series_ds[:last_train_idx]

test_y = time_series[last_train_idx:]
test_ds = time_series_ds[last_train_idx:]

# plot time series with train / test split borders as vertical lines
plt.plot(time_series)
plt.axvline(x=last_train_idx, color="r", label="train / test split", linestyle="--")
plt.legend()
plt.title("Time Series")
plt.show()

nf_ti_adapter.fit(train_ds, train_y, val_size=int(VAL_SPLIT * len(train_y)))

# pick random INPUT_SIZE window from test set
start_idx = random.randint(0, len(test_ds) - INPUT_SIZE)
test_input_ds = test_ds[start_idx : start_idx + INPUT_SIZE]
test_input_y = test_y[start_idx : start_idx + INPUT_SIZE]

predictions = nf_ti_adapter.predict(
    ds=test_input_ds, y=test_input_y  # , test_ds=test_ds, test_y=test_y
)

target_indices = list(range(len(predictions[f"{model}-loc"])))
attribution_list = nf_ti_adapter.explain("TIG", target_indices, "-loc")

for idx, attributions in enumerate(attribution_list):
    target_idx = target_indices[idx]
    for i, attr in enumerate(attributions):
        plt.axvline(
            x=float(test_input_ds[i]),
            color="r",
            alpha=attr,
            linewidth=3,
        )
    plt.plot(test_input_ds, test_input_y)
    plt.plot(predictions.ds, predictions[f"{model}-loc"])
    plt.fill_between(
        predictions.ds,
        np.subtract(predictions[f"{model}-loc"], predictions[f"{model}-scale"]),
        np.add(predictions[f"{model}-loc"], predictions[f"{model}-scale"]),
        alpha=0.2,
        color="green",
        label="standard deviation",
    )
    plt.scatter(
        predictions.ds.iloc[target_idx],
        predictions[f"{model}-loc"].iloc[target_idx],
        color="green",
        label="predicted point",
        marker="x",
    )
    plt.title("Attributions for predicted point")
    plt.legend()
    plt.show()
