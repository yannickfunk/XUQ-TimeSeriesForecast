import os

import numpy as np
from matplotlib import pyplot as plt
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM
from synthetictime.simple_time_series import SimpleTimeSeries

from nf_ti_adapter.lstm import LstmNfTiAdapter

os.environ["NIXTLA_ID_AS_COL"] = "1"

HORIZON = 64
LEVELS = [80, 90]
INPUT_SIZE = 2 * HORIZON

model = LSTM(
    input_size=INPUT_SIZE,
    inference_input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=500,
)

nf_ti_adapter = LstmNfTiAdapter(model, 1)
ts = SimpleTimeSeries(
    size=1000,
    base_amplitude=2.0,
    base_frequency=0.03,
    base_noise_scale=0,
    base_noise_amplitude=0.5,
)

train_y = ts.synthetic_time_series[:-HORIZON]
train_ds = np.arange(len(train_y))

# add noise to all values > 1.5 in train_y
train_y[train_y > 1] += np.random.normal(0, 1, len(train_y[train_y > 1]))

test_y = ts.synthetic_time_series[-HORIZON:]
test_ds = np.arange(len(train_y), len(train_y) + len(test_y))

nf_ti_adapter.fit(train_ds, train_y)

predictions = nf_ti_adapter.predict_plot(test_ds=test_ds, test_y=test_y)

plt.plot(predictions.ds, predictions["LSTM-scale"])
plt.show()

target_indices = np.argwhere(predictions["LSTM-scale"] > 1).flatten().tolist()
attribution_list = nf_ti_adapter.explain("TIG", target_indices, "-loc")

attributions = np.mean(attribution_list, axis=0)

plt.plot(train_ds[-INPUT_SIZE:], train_y[-INPUT_SIZE:])
for i, attr in enumerate(attributions[-INPUT_SIZE:]):
    xmin = train_ds[-INPUT_SIZE + i]
    xmax = train_ds[-INPUT_SIZE + i + 1]

    if xmin > xmax:
        continue
    plt.axvspan(
        xmin=xmin,
        xmax=xmax,
        color="r",
        alpha=attr,
    )
plt.show()
