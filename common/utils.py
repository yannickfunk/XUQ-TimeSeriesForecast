import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from scipy.signal import find_peaks
from synthetictime.simple_time_series import SimpleTimeSeries

from common.timeseries import AttributedTimeSeries, TimeSeries


def plot_attributions(
    attribution_list,
    negative_attribution_list,
    raw_attribution_list,
    raw_negative_attribution_list,
    output_name,
    test_input_ds,
    test_input_y,
    predictions,
    model,
    method,
):
    # create results/{method} if it does not exist
    os.makedirs(f"results/{method}", exist_ok=True)
    os.makedirs(f"results_tikz/{method}", exist_ok=True)
    os.makedirs(f"results_csv/{method}", exist_ok=True)

    for idx, attributions in enumerate(attribution_list):
        target_idx = idx

        df = pd.DataFrame(
            columns=["ds", "y", "pos_attr", "neg_attr", "raw_pos_attr", "raw_neg_attr"]
        )
        df["ds"] = test_input_ds
        df["y"] = test_input_y
        df["pos_attr"] = attributions
        df["neg_attr"] = negative_attribution_list[idx]
        df["raw_pos_attr"] = raw_attribution_list[idx]
        df["raw_neg_attr"] = raw_negative_attribution_list[idx]
        df.to_csv(
            f"results_csv/{method}/{output_name[1:]}_attributions_{target_idx}.csv"
        )

        predictions_df = predictions[["ds", f"{model}-loc", f"{model}-scale"]]
        predictions_df.to_csv(f"results_csv/predictions.csv")

        for i, attr in enumerate(attributions):
            if not isinstance(test_input_ds[i], np.datetime64):
                x_value = float(test_input_ds[i])
            else:
                x_value = test_input_ds[i]

            plt.axvline(
                x=x_value,
                color="r",
                alpha=attr,
                linewidth=3,
            )
        for i, attr in enumerate(negative_attribution_list[idx]):
            if not isinstance(test_input_ds[i], np.datetime64):
                x_value = float(test_input_ds[i])
            else:
                x_value = test_input_ds[i]

            plt.axvline(
                x=x_value,
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
        plt.savefig(f"results/{method}/{output_name[1:]}_attributions_{target_idx}.png")
        tikzplotlib.save(
            f"results_tikz/{method}/{output_name[1:]}_attributions_{target_idx}.tex"
        )
        plt.close()
        # plt.show()


def plot_attributions_exogenous(
    attributed_time_series_list: List[AttributedTimeSeries],
    target_uid: str,
    output_name,
    predictions,
    model,
    method,
):
    # create results/{method} if it does not exist
    os.makedirs(f"results/{method}", exist_ok=True)
    os.makedirs(f"results_tikz/{method}", exist_ok=True)
    os.makedirs(f"results_csv/{method}", exist_ok=True)

    for target_idx in range(len(attributed_time_series_list[0].positive_attributions)):
        df = pd.DataFrame()

        for ats in attributed_time_series_list:

            if ats.unique_id == target_uid:
                df["ds"] = ats.ds
                df["y"] = ats.y
                df["pos_attr_y"] = ats.positive_attributions[target_idx]
                df["neg_attr_y"] = ats.negative_attributions[target_idx]
            else:
                df[ats.unique_id] = ats.y
                df[f"pos_attr_{ats.unique_id}"] = ats.positive_attributions[target_idx]
                df[f"neg_attr_{ats.unique_id}"] = ats.negative_attributions[target_idx]

        df.to_csv(
            f"results_csv/{method}/{output_name[1:]}_attributions_{target_idx}.csv"
        )

        predictions_df = predictions[["ds", f"{model}-loc", f"{model}-scale"]]
        predictions_df.to_csv(f"results_csv/predictions.csv")

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
                if f"{model}-loc" in predictions.columns:
                    variance = predictions[f"{model}-scale"]
                    median = predictions[f"{model}-loc"]
                else:
                    median = predictions[f"{model}-median"]
                    variance = (
                        predictions[f"{model}-hi-68"].values
                        - predictions[f"{model}-lo-68"].values
                    ) / 2

                axs[i].fill_between(
                    predictions.ds,
                    np.subtract(
                        median,
                        variance,
                    ),
                    np.add(
                        median,
                        variance,
                    ),
                    alpha=0.2,
                    color="green",
                    label="standard deviation",
                )
                axs[i].plot(
                    predictions.ds,
                    median,
                    label="predictions",
                    color="green",
                )
                axs[i].plot(
                    predictions.ds.iloc[target_idx],
                    median.iloc[target_idx],
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
        plt.savefig(f"results/{method}/{output_name[1:]}_attributions_{target_idx}.png")
        tikzplotlib.save(
            f"results_tikz/{method}/{output_name[1:]}_attributions_{target_idx}.tex"
        )
        plt.close()
        # plt.show()
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
        if end > len(array):
            break

        array[start:end] += np.random.normal(noise_mean, noise_std, size=end - start)
    return array


def add_amplitude_random(array):
    # get indices where array is close to zero
    zero_indices = np.argwhere(np.abs(array) < 0.2).T[0]

    i = 0
    indices_factors = []
    while i < len(zero_indices) - 2:
        idx = zero_indices[i]
        next_idx = zero_indices[i + 2] if i + 2 < len(zero_indices) else len(array)
        length = next_idx - idx

        if random.random() < 0.1:
            factor = 3 * np.ones(length)
            array[idx:next_idx] *= factor
            indices_factors.append(((idx, next_idx), factor[0]))
        i += 2
    return array, indices_factors


def add_amplitude(array, indices_factors, shift):
    for (start, end), factor in indices_factors:
        # check if shift exceeds bounds
        if end + shift > len(array):
            break

        array[start + shift : end + shift] *= factor
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


def generate_sine(
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
    return base


def add_spiky_peaks(array):
    # also return indices
    peaks, _ = find_peaks(array)
    spike_idx = []
    for peak in peaks:
        # skip peak with a probability
        if random.random() < 0.8:
            continue
        array[peak] += -5
        spike_idx.append(peak)
    return array, np.array(spike_idx)[:-1]


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
