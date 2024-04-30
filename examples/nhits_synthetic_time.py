import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from synthetictime.simple_time_series import SimpleTimeSeries

# First we set a random seed
np.random.seed(7)

# Now we initiate a simple time series object with the desired parameters
# This also automatically generates the time series
example_series = SimpleTimeSeries(
    size=200,
    base_amplitude=5.0,
    base_frequency=0.05,
    base_noise_scale=1,
    base_noise_amplitude=1,
    number_of_trend_events=2,
    trend_slope_low=0.01,
    trend_slope_high=0.2,
    number_of_cosine_events=3,
    cosine_frequency_low=0.01,
    cosine_frequency_high=0.2,
    cosine_amplitude_low=2.0,
    cosine_amplitude_high=8,
    number_of_increased_noise_events=2,
    increased_noise_low=0.8,
    increased_noise_high=2,
    noise_type="additive",
)
synthetic_timeseries = example_series.synthetic_time_series

# transform the synthetic time series into a pandas dataframe
synthetic_timeseries = pd.DataFrame(
    {
        "unique_id": 1.0,
        "ds": pd.date_range(
            start="2020-01-01", freq="D", periods=len(synthetic_timeseries)
        ),
        "y": synthetic_timeseries,
    }
)


Y_train = synthetic_timeseries[:128]
Y_test = synthetic_timeseries[128:]
horizon = len(Y_test)

nf = NeuralForecast(
    models=[NHITS(input_size=8, h=horizon, start_padding_enabled=True)],
    freq="D",
)
nf.fit(df=Y_train)
predictions = nf.predict()


plt.plot(synthetic_timeseries.ds, synthetic_timeseries.y, label="Ground Truth")
plt.plot(predictions.ds, predictions.NHITS, label="Predictions")
plt.legend()
plt.show()
