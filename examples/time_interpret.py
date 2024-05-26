from typing import Tuple

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM
from synthetictime.simple_time_series import SimpleTimeSeries
from tint.attr import TemporalIntegratedGradients

# First we set a random seed
np.random.seed(7)


HORIZON = 86
LEVELS = [80, 90]
INPUT_SIZE = 3 * HORIZON


def get_synthetic_series() -> pd.DataFrame:
    ts = SimpleTimeSeries(
        size=1000,
        base_amplitude=2.0,
        base_frequency=0.02,
        base_noise_scale=1,
        base_noise_amplitude=0.5,
    )

    # transform the synthetic time series into a pandas dataframe
    synthetic_timeseries = pd.DataFrame(
        {
            "unique_id": 1.0,
            "ds": np.arange(len(ts.synthetic_time_series)),
            "y": ts.synthetic_time_series,
        }
    )
    return synthetic_timeseries


def train_test_split(
    synthetic_timeseries: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_train = synthetic_timeseries[:-HORIZON]
    y_test = synthetic_timeseries[-HORIZON:]
    return y_train, y_test


def define_neural_forecast() -> NeuralForecast:
    models = [
        LSTM(
            input_size=INPUT_SIZE,
            h=HORIZON,
            loss=DistributionLoss(
                distribution="Normal", level=LEVELS, return_params=True
            ),
            max_steps=500,
        )
    ]
    nf = NeuralForecast(
        models=models,
        freq=1,
    )
    return nf


def plot_predictions_quantile(
    nf: NeuralForecast, synthetic_timeseries: pd.DataFrame, predictions: pd.DataFrame
):
    model = nf.models[0]
    last = HORIZON * 3
    plt.plot(
        synthetic_timeseries.ds.iloc[-last:],
        synthetic_timeseries.y.iloc[-last:],
        label="ground truth",
        alpha=0.5,
    )
    plt.plot(
        predictions.ds,
        predictions[f"{model}-median"],
        label="median prediction",
        color="green",
    )

    for i, level in enumerate(LEVELS):
        plt.fill_between(
            predictions.ds,
            predictions[f"{model}-lo-{level}"],
            predictions[f"{model}-hi-{level}"],
            alpha=1 - (i + 1) * (1 / (len(LEVELS) + 1)),
            color="orange",
            label=f"{level}% prediction interval",
        )
    plt.title(f"Predictions with prediction intervals, model: {model}")
    plt.legend(loc="upper left")
    plt.show()


def plot_predictions_parametric(
    nf: NeuralForecast, synthetic_timeseries: pd.DataFrame, predictions: pd.DataFrame
):
    model = nf.models[0]
    last = HORIZON * 3
    plt.plot(
        synthetic_timeseries.ds.iloc[-last:],
        synthetic_timeseries.y.iloc[-last:],
        label="ground truth",
        alpha=0.5,
    )
    plt.plot(
        predictions.ds,
        predictions[f"{model}-loc"],
        label="mean prediction",
        color="green",
    )
    plt.fill_between(
        predictions.ds,
        np.subtract(predictions[f"{model}-loc"], predictions[f"{model}-scale"]),
        np.add(predictions[f"{model}-loc"], predictions[f"{model}-scale"]),
        alpha=0.2,
        color="green",
        label="standard deviation",
    )

    plt.title(f"Mean prediction and standard deviation, model: {model}")
    plt.legend(loc="upper left")
    plt.show()


def forward_function(model: pl.LightningModule, inputs: torch.Tensor):
    inputs = inputs[0, :, 0]
    masks = torch.ones_like(inputs)
    inputs = torch.unsqueeze(torch.vstack([inputs, masks]), 0)
    batch = {
        "temporal": inputs,
        "temporal_cols": pd.Index(["y", "available_mask"]),
        "y_idx": 0,
    }
    batch_idx = 0
    model_output = model.predict_step(batch, batch_idx)
    return model_output[0, -1, :, -1]


def main():
    synthetic_timeseries = get_synthetic_series()

    plt.plot(synthetic_timeseries.ds, synthetic_timeseries.y)
    plt.show()

    y_train, y_test = train_test_split(synthetic_timeseries)

    nf = define_neural_forecast()
    nf.fit(df=y_train)

    predictions = nf.predict()
    plot_predictions_quantile(nf, synthetic_timeseries, predictions)
    plot_predictions_parametric(nf, synthetic_timeseries, predictions)

    model = nf.models[0]
    inputs = torch.from_numpy(np.expand_dims(np.array(y_train.y), axis=(0, -1))).float()
    raw_predictions = forward_function(model, inputs)

    # plot raw predictions vs predictions
    plt.plot(predictions.ds, predictions["LSTM-scale"])
    plt.plot(predictions.ds, raw_predictions.detach().numpy())
    plt.title("Check for matching predictions")
    plt.show()

    explainer = TemporalIntegratedGradients(lambda x: forward_function(model, x))

    attr = explainer.attribute(inputs).abs()

    plt.plot(y_train.ds.iloc[-INPUT_SIZE:], y_train.y.iloc[-INPUT_SIZE:])
    plt.plot(y_train.ds.iloc[-INPUT_SIZE:], attr.squeeze().numpy()[-INPUT_SIZE:] * 10)
    plt.show()


if __name__ == "__main__":
    main()
