import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from scipy.signal import find_peaks
from synthetictime.simple_time_series import SimpleTimeSeries

from common.timeseries import TimeSeries


def plot_attributions(
    attribution_list,
    negative_attribution_list,
    output_name,
    test_input_ds,
    test_input_y,
    predictions,
    model,
    method,
):
    for idx, attributions in enumerate(attribution_list):
        target_idx = idx
        for i, attr in enumerate(attributions):
            plt.axvline(
                x=float(test_input_ds[i]),
                color="r",
                alpha=attr,
                linewidth=3,
            )
        for i, attr in enumerate(negative_attribution_list[idx]):
            plt.axvline(
                x=float(test_input_ds[i]),
                color="b",
                alpha=attr,
                linewidth=3,
            )
        plt.plot(test_input_ds, test_input_y, color="black")
        plt.plot(predictions.ds, predictions[f"{model}-loc"])
        plt.fill_between(
            predictions.ds,
            np.subtract(predictions[f"{model}-loc"], predictions[f"{model}-scale"]),
            np.add(predictions[f"{model}-loc"], predictions[f"{model}-scale"]),
            alpha=0.2,
            color="green",
            label="standard deviation",
        )
        plt.plot(
            predictions.ds.iloc[target_idx],
            predictions[f"{model}-loc"].iloc[target_idx],
            color="green",
            label="predicted point",
            linestyle=None,
            marker="x",
        )
        plt.title(f"{method} Attributions for {output_name}, predictions with {model}")
        plt.legend()
        plt.savefig(f"results/{output_name[1:]}_attributions_{target_idx}.png")
        tikzplotlib.save(
            f"results_tikz/{output_name[1:]}_attributions_{target_idx}.tex"
        )
        plt.show()


def generate_sine_noisy_peaks(
    size: int = 1000,
    amplitude: float = 2.0,
    frequency: float = 0.05,
    noise_mean: float = 0,
    noise_std: float = 2,
    noise_width: int = 6,
):
    base = SimpleTimeSeries(
        size=size,
        base_amplitude=amplitude,
        base_frequency=frequency,
        base_noise_scale=0,
        base_noise_amplitude=0,
    ).synthetic_time_series

    peaks, _ = find_peaks(base)
    for peak in peaks:
        # skip peak with a probability
        if random.random() < 0.8:
            continue
        limit = noise_width // 2
        sub_arr = base[peak - limit : peak + limit + 1]
        sub_arr += np.random.normal(noise_mean, noise_std, len(sub_arr))
    return base


def train_test_split(
    time_series_list: List[TimeSeries], train_split
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    last_train_idx = int(len(time_series_list[0].y) * train_split)
    train = [
        TimeSeries(
            unique_id=ts.unique_id,
            ds=ts.ds[:last_train_idx],
            y=ts.y[:last_train_idx],
        )
        for ts in time_series_list
    ]
    test = [
        TimeSeries(
            unique_id=ts.unique_id,
            ds=ts.ds[last_train_idx:],
            y=ts.y[last_train_idx:],
        )
        for ts in time_series_list
    ]
    return train, test
