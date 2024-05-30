import os

import numpy as np
from matplotlib import pyplot as plt
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM
from synthetictime.simple_time_series import SimpleTimeSeries

from nf_ti_adapter.lstm import LstmNfTiAdapter

os.environ["NIXTLA_ID_AS_COL"] = "1"

HORIZON = 86
LEVELS = [80, 90]
INPUT_SIZE = 3 * HORIZON

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
    base_frequency=0.02,
    base_noise_scale=1,
    base_noise_amplitude=0.5,
)
train_y = ts.synthetic_time_series[:-HORIZON]
train_ds = np.arange(len(train_y))

test_y = ts.synthetic_time_series[-HORIZON:]
test_ds = np.arange(len(train_y), len(train_y) + len(test_y))

nf_ti_adapter.fit(train_ds, train_y)

nf_ti_adapter.predict_plot(test_ds=test_ds, test_y=test_y)

attributions = nf_ti_adapter.explain("TIG", [0])

plt.plot(train_ds[-INPUT_SIZE:], train_y[-INPUT_SIZE:])
plt.plot(train_ds[-INPUT_SIZE:], attributions[0].squeeze().numpy()[-INPUT_SIZE:] * 10)
plt.show()
