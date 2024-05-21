from typing import Tuple

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from synthetictime.simple_time_series import SimpleTimeSeries
from tint.attr import TemporalIntegratedGradients

# from neuralforecast.losses.pytorch import MQLoss
# from lightning.pytorch.utilities.model_summary import ModelSummary
# First we set a random seed
np.random.seed(7)


HORIZON = 86
LEVELS = [60, 99]
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
            h=HORIZON,  # loss=MQLoss(level=LEVELS),
            max_steps=1000,
        )
    ]
    nf = NeuralForecast(
        models=models,
        freq=1,
    )
    return nf


def plot_predictions(
    nf: NeuralForecast, synthetic_timeseries: pd.DataFrame, predictions: pd.DataFrame
):
    model = nf.models[0]
    last = HORIZON * 3
    plt.plot(
        synthetic_timeseries.ds.iloc[-last:],
        synthetic_timeseries.y.iloc[-last:],
        label="Ground Truth",
        alpha=0.5,
    )
    plt.plot(
        predictions.ds,
        predictions[f"{model}"],
        label="Predictions",
        color="green",
    )
    # plt.plot(
    #     predictions.ds.iloc[-last:],
    #     predictions[f"{model}-median"].iloc[-last:],
    #     label="Predictions",
    #     color="green",
    # )
    #
    # for i, level in enumerate(LEVELS):
    #     plt.fill_between(
    #         predictions.ds.iloc[-last:],
    #         predictions[f"{model}-lo-{level}"].iloc[-last:],
    #         predictions[f"{model}-hi-{level}"].iloc[-last:],
    #         alpha=1 - (i + 1) * (1 / (len(LEVELS) + 1)),
    #         color="orange",
    #         label=f"Level {level}",
    #     )

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
    return model_output[0, -1]


def main():
    synthetic_timeseries = get_synthetic_series()

    plt.plot(synthetic_timeseries.ds, synthetic_timeseries.y)
    plt.show()

    y_train, y_test = train_test_split(synthetic_timeseries)

    nf = define_neural_forecast()
    nf.fit(df=y_train)

    predictions = nf.predict()
    plot_predictions(nf, synthetic_timeseries, predictions)
    model = nf.models[0]

    inputs = torch.from_numpy(np.expand_dims(np.array(y_train.y), axis=(0, -1))).float()
    raw_predictions = forward_function(model, inputs)

    # plot raw predictions vs predictions
    plt.plot(predictions.ds, predictions["LSTM"])
    plt.plot(predictions.ds, raw_predictions.detach().numpy())
    plt.show()

    explainer = TemporalIntegratedGradients(lambda x: forward_function(model, x))

    attr = explainer.attribute(inputs).abs()

    plt.plot(y_train.ds.iloc[-INPUT_SIZE:], y_train.y.iloc[-INPUT_SIZE:])
    plt.plot(y_train.ds.iloc[-INPUT_SIZE:], attr.squeeze().numpy()[-INPUT_SIZE:] * 10)
    plt.show()


if __name__ == "__main__":
    main()
