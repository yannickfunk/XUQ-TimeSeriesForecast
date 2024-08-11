import os

import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from neuralforecast.models import NHITS

from common.timeseries import TimeSeries
from common.utils import (add_noise, generate_sine_noisy_peaks,
                          generate_sine_spiky_peaks,
                          plot_attributions_exogenous, plot_time_series_list,
                          train_test_split)
from nf_ti_adapter.nhits import NhitsNfTiAdapter

os.environ["NIXTLA_ID_AS_COL"] = "1"

SIZE = 64000
HORIZON = 48
LEVELS = [68]
INPUT_SIZE = 2 * HORIZON
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

model = NHITS(
    input_size=INPUT_SIZE,
    h=HORIZON,
    # loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    loss=MQLoss(level=LEVELS),
    max_steps=1700,
    random_seed=40,
    # early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
    hist_exog_list=["sine_spiky_a", "sine_spiky_b"],
    scaler_type="identity",
    # exclude_insample_y=True,
)

nf_ti_adapter = NhitsNfTiAdapter(model, 1)

shift = 40
_time_series_spiky_a, spike_idx_a = generate_sine_spiky_peaks(size=SIZE)
_time_series_spiky_b, spike_idx_b = generate_sine_spiky_peaks(size=SIZE)

_time_series_noisy = generate_sine_noisy_peaks(size=SIZE, noise_std=0)
_time_series_noisy = add_noise(_time_series_noisy, spike_idx_a + 40)
_time_series_noisy = add_noise(_time_series_noisy, spike_idx_b + 40)

_time_series_ds = np.arange(len(_time_series_spiky_a))

time_series_list = [
    TimeSeries(
        unique_id="sine_spiky_a",
        ds=_time_series_ds,
        y=_time_series_spiky_a,
    ),
    TimeSeries(
        unique_id="sine_spiky_b",
        ds=_time_series_ds,
        y=_time_series_spiky_b,
    ),
    TimeSeries(
        unique_id="sine_noisy",
        ds=_time_series_ds,
        y=_time_series_noisy,
    ),
]

# train test split
train_time_series_list, test_time_series_list = train_test_split(
    time_series_list, TRAIN_SPLIT
)

# plot time series list with train / test split borders as vertical lines
plot_time_series_list(train_time_series_list, limit=(0, 200))

start_idx = 1420
test_input_list = [
    TimeSeries(
        unique_id=ts.unique_id,
        ds=ts.ds[start_idx : start_idx + INPUT_SIZE],
        y=ts.y[start_idx : start_idx + INPUT_SIZE],
    )
    for ts in test_time_series_list
]
plot_time_series_list(test_input_list)

nf_ti_adapter.fit_list_exogenous(
    train_time_series_list,
    "sine_noisy",
    val_size=int(VAL_SPLIT * len(train_time_series_list[0].y)),
)
predictions = nf_ti_adapter.predict_list_exogenous_plot(
    test_input_list, target_uid="sine_noisy"
)

target_indices = list(range(HORIZON))
attributed_timeseries_list = nf_ti_adapter.explain_list(
    "TIG", target_indices, "-median", "sine_noisy", test_input_list
)
plot_attributions_exogenous(
    attributed_timeseries_list,
    "sine_noisy",
    "-loc",
    predictions,
    model,
    "TIG",
)

target_indices = list(range(HORIZON))
attributed_timeseries_list = nf_ti_adapter.explain_list(
    "TIG", target_indices, "-lo-68", "sine_noisy", test_input_list
)
plot_attributions_exogenous(
    attributed_timeseries_list,
    "sine_noisy",
    "-scale",
    predictions,
    model,
    "TIG",
)
