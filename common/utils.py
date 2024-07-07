import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


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
