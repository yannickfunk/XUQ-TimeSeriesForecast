import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from scipy.signal import find_peaks
from synthetictime.simple_time_series import SimpleTimeSeries

from common.timeseries import AttributedTimeSeries, TimeSeries


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


def plot_attributions_exogenous(
    attributed_time_series_list: List[AttributedTimeSeries],
    target_uid: str,
    output_name,
    predictions,
    model,
    method,
):

    for target_idx in range(len(attributed_time_series_list[0].positive_attributions)):
        # plot time series list with train / test split borders as vertical lines
        fig, axs = plt.subplots(len(attributed_time_series_list), 1, sharex="all")

        for i, ts in enumerate(attributed_time_series_list):
            axs[i].plot(ts.ds, ts.y)

            for attr_idx, attr in enumerate(ts.positive_attributions[target_idx]):
                axs[i].axvline(
                    x=float(ts.ds[attr_idx]),
                    color="r",
                    alpha=attr,
                    linewidth=3,
                )
            for attr_idx, attr in enumerate(ts.negative_attributions[target_idx]):
                axs[i].axvline(
                    x=float(ts.ds[attr_idx]),
                    color="b",
                    alpha=attr,
                    linewidth=3,
                )

            if ts.unique_id == target_uid:
                axs[i].plot(
                    predictions.ds,
                    predictions[f"{model}-loc"],
                    label="predictions",
                    color="green",
                )
                axs[i].fill_between(
                    predictions.ds,
                    np.subtract(
                        predictions[f"{model}-loc"],
                        predictions[f"{model}-scale"],
                    ),
                    np.add(
                        predictions[f"{model}-loc"],
                        predictions[f"{model}-scale"],
                    ),
                    alpha=0.2,
                    color="green",
                    label="standard deviation",
                )
                axs[i].plot(
                    predictions.ds.iloc[target_idx],
                    predictions[f"{model}-loc"].iloc[target_idx],
                    color="green",
                    label="predicted point",
                    linestyle=None,
                    marker="x",
                    markersize=10,
                )
                axs[i].set_title(f"{ts.unique_id} - predictions")
            else:
                axs[i].set_title(ts.unique_id)

            axs[i].set_ylim(-6, 6)
            fig.subplots_adjust(hspace=0.6)

        fig.suptitle(
            f"{method} Attributions for {output_name}, predictions with {model}"
        )
        plt.savefig(f"results/{output_name[1:]}_attributions_{target_idx}.png")
        tikzplotlib.save(
            f"results_tikz/{output_name[1:]}_attributions_{target_idx}.tex"
        )
        plt.show()
    return predictions


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


def add_noise(
    array,
    indices,
    noise_mean: float = 0,
    noise_std: float = 2,
    noise_width: int = 6,
):
    for idx in indices:
        limit = noise_width // 2
        start = idx - limit
        end = idx + limit + 1
        array[start:end] += np.random.normal(noise_mean, noise_std, size=end - start)
    return array


def generate_sine_spiky(
    size: int = 1000,
    amplitude: float = 2.0,
    frequency: float = 0.05,
):
    base = SimpleTimeSeries(
        size=size,
        base_amplitude=amplitude,
        base_frequency=frequency,
        base_noise_scale=0,
        base_noise_amplitude=0,
    ).synthetic_time_series
    spike_idx = []
    for i in range(len(base)):
        if random.random() < 0.97:
            continue
        base[i] += -5
        spike_idx.append(i)
    return base, spike_idx


def generate_sine_spiky_peaks(
    size: int = 1000,
    amplitude: float = 2.0,
    frequency: float = 0.05,
):
    base = SimpleTimeSeries(
        size=size,
        base_amplitude=amplitude,
        base_frequency=frequency,
        base_noise_scale=0,
        base_noise_amplitude=0,
    ).synthetic_time_series
    peaks, _ = find_peaks(base)
    spike_idx = []
    for peak in peaks:
        # skip peak with a probability
        if random.random() < 0.8:
            continue
        base[peak] += -5
        spike_idx.append(peak)
    return base, np.array(spike_idx)[:-1]


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


def plot_time_series_list(time_series_list: List[TimeSeries], limit=None):
    fig, axs = plt.subplots(len(time_series_list), 1, sharex="all")
    for i, time_series in enumerate(time_series_list):
        if limit:
            axs[i].plot(time_series.y[limit[0] : limit[1]])
        else:
            axs[i].plot(time_series.y)
        axs[i].set_ylim(-5, 5)
        axs[i].set_title(time_series.unique_id)
        fig.subplots_adjust(hspace=0.6)
    plt.show()
    return fig, axs
