import os

import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from neuralforecast.models import NHITS

from common.timeseries import TimeSeries
from common.utils import (add_amplitude, add_amplitude_random, add_noise,
                          add_spiky_peaks, generate_sine,
                          plot_attributions_exogenous, plot_time_series_list,
                          train_test_split)
from nf_ti_adapter.base import METHOD_TO_CONSTRUCTOR
from nf_ti_adapter.nhits import NhitsNfTiAdapter

ATTR_METHODS = METHOD_TO_CONSTRUCTOR.keys()

os.environ["NIXTLA_ID_AS_COL"] = "1"

SIZE = 100000
HORIZON = 48
LEVELS = [68]
INPUT_SIZE = 2 * HORIZON
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

model = NHITS(
    input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=[80, 90], return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=600,
    random_seed=40,
    # early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
    hist_exog_list=["sine_spiky_a", "sine_spiky_b"],
    scaler_type="robust",
    # exclude_insample_y=True,
)

nf_ti_adapter = NhitsNfTiAdapter(model, 1)

shift = 40
_time_series_spiky_a = generate_sine(size=SIZE)
_time_series_spiky_a, amp_idx_a = add_amplitude_random(_time_series_spiky_a)
_time_series_spiky_a, spike_idx_a = add_spiky_peaks(_time_series_spiky_a)

_time_series_spiky_b = generate_sine(size=SIZE)
_time_series_spiky_b, amp_idx_b = add_amplitude_random(_time_series_spiky_b)
_time_series_spiky_b, spike_idx_b = add_spiky_peaks(_time_series_spiky_b)

_time_series_noisy = generate_sine(size=SIZE)

_time_series_noisy = add_noise(_time_series_noisy, spike_idx_a + shift)
_time_series_noisy = add_noise(_time_series_noisy, spike_idx_b + shift)

_time_series_noisy = add_amplitude(_time_series_noisy, amp_idx_a, shift)
_time_series_noisy = add_amplitude(_time_series_noisy, amp_idx_b, shift)

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

start_idx = 96 + 40
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

for attr_method in ATTR_METHODS:
    target_indices = list(range(HORIZON))
    attributed_timeseries_list = nf_ti_adapter.explain_list(
        attr_method, target_indices, "-loc", "sine_noisy", test_input_list
    )
    plot_attributions_exogenous(
        attributed_timeseries_list,
        "sine_noisy",
        "-loc",
        predictions,
        model,
        attr_method,
    )

    target_indices = list(range(HORIZON))
    attributed_timeseries_list = nf_ti_adapter.explain_list(
        attr_method, target_indices, "-scale", "sine_noisy", test_input_list
    )
    plot_attributions_exogenous(
        attributed_timeseries_list,
        "sine_noisy",
        "-scale",
        predictions,
        model,
        attr_method,
    )
