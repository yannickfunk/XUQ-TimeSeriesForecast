from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

plt.style.use("ggplot")

EXP_NAME = "exogenous_sinus_peak_spike"
exp_root = Path("../experimental_runs")

model_to_output = {
    "mlp": "MLP",
    "nbeats": "NBEATSx",
    "nhits": "NHITS",
    "patch_tst": "PatchTST",
}

model_to_display = {
    "mlp": "MLP",
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "patch_tst": "PatchTST",
}

for exp_model in ["mlp", "nbeats", "nhits"]:
    exp_path = exp_root / EXP_NAME / exp_model
    csv_results_path = exp_path / "results_csv"
    predictions_path = csv_results_path / "predictions.csv"

    for exp_attr in ["FA", "IG", "IXG", "LIME"]:
        for output in ["loc", "scale"]:
            fa_path = csv_results_path / exp_attr
            predictions_df = pd.read_csv(predictions_path)

            fig, axes = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(12, 5))
            fig.subplots_adjust(wspace=0.08, hspace=0.3)

            target_indices = [10, 20, 30, 35]

            for ax, target_index in zip(axes.flatten(), target_indices):
                # get top bottom right and left of ax
                top = ax.get_position().get_points()[1][1]
                bottom = ax.get_position().get_points()[0][1]
                right = ax.get_position().get_points()[1][0]
                left = ax.get_position().get_points()[0][0]
                ax.axis("off")

                gs = gridspec.GridSpec(
                    3,
                    1,
                    figure=fig,
                    top=top,
                    bottom=bottom,
                    right=right,
                    left=left,
                    hspace=0.05,
                )
                sub_ax0 = fig.add_subplot(gs[0])
                sub_ax1 = fig.add_subplot(gs[1])
                sub_ax2 = fig.add_subplot(gs[2])

                csv_path = fa_path / f"{output}_attributions_{target_index}.csv"
                df = pd.read_csv(csv_path)

                input_x = np.arange(len(df["ds"]))
                prediction_x = np.arange(
                    len(df["ds"]), len(df["ds"]) + len(predictions_df)
                )

                y = df["y"]

                pos_attr_y = df["pos_attr_y"]
                neg_attr_y = df["neg_attr_y"]

                sine_a = df["sine_spiky_a"]
                pos_attr_sine_a = df["pos_attr_sine_spiky_a"]
                neg_attr_sine_a = df["neg_attr_sine_spiky_a"]

                sine_b = df["sine_spiky_b"]
                pos_attr_sine_b = df["pos_attr_sine_spiky_b"]
                neg_attr_sine_b = df["neg_attr_sine_spiky_b"]

                for i in range(len(pos_attr_y)):
                    sub_ax0.axvline(
                        x=input_x[i], color="red", alpha=pos_attr_y.iloc[i], lw=3
                    )
                    sub_ax0.axvline(
                        x=input_x[i], color="blue", alpha=neg_attr_y.iloc[i], lw=3
                    )

                for i in range(len(pos_attr_sine_a)):
                    sub_ax1.axvline(
                        x=input_x[i], color="red", alpha=pos_attr_sine_a.iloc[i], lw=3
                    )
                    sub_ax1.axvline(
                        x=input_x[i], color="blue", alpha=neg_attr_sine_a.iloc[i], lw=3
                    )

                for i in range(len(pos_attr_sine_b)):
                    sub_ax2.axvline(
                        x=input_x[i], color="red", alpha=pos_attr_sine_b.iloc[i], lw=3
                    )
                    sub_ax2.axvline(
                        x=input_x[i], color="blue", alpha=neg_attr_sine_b.iloc[i], lw=3
                    )

                sub_ax0.plot(input_x, y, color="black", label="Input")
                sub_ax1.plot(input_x, sine_a, color="black", label="Input")
                sub_ax2.plot(input_x, sine_b, color="black", label="Input")

                # Plot the predictions
                sub_ax0.plot(
                    prediction_x,
                    predictions_df[f"{model_to_output[exp_model]}-loc"],
                    label=f"Mean Prediction",
                    color="green",
                )

                # Fill between for the variance
                sub_ax0.fill_between(
                    prediction_x,
                    predictions_df[f"{model_to_output[exp_model]}-loc"]
                    - predictions_df[f"{model_to_output[exp_model]}-scale"],
                    predictions_df[f"{model_to_output[exp_model]}-loc"]
                    + predictions_df[f"{model_to_output[exp_model]}-scale"],
                    label=f"Std. Dev. Prediction",
                    color="green",
                    alpha=0.2,
                )

                sub_ax0.plot(
                    prediction_x[target_index],
                    predictions_df[f"{model_to_output[exp_model]}-loc"].iloc[
                        target_index
                    ],
                    color="black",
                    marker="x",
                    markersize=8,
                    markeredgewidth=2,
                    label="Explained Point",
                    linestyle="None",
                )

                sub_ax0.set_ylim(-5, 5)
                sub_ax1.set_ylim(-5, 5)
                sub_ax2.set_ylim(-5, 5)

                sub_ax0.set_yticks([-3, 0, 3])
                sub_ax1.set_yticks([-3, 0, 3])
                sub_ax2.set_yticks([-3, 0, 3])

                sub_ax0.set_xlim(0, 144)
                sub_ax1.set_xlim(0, 144)
                sub_ax2.set_xlim(0, 144)

                sub_ax0.tick_params(axis="x", labelsize=16)
                sub_ax0.tick_params(axis="y", labelsize=11)

                sub_ax1.tick_params(axis="x", labelsize=16)
                sub_ax1.tick_params(axis="y", labelsize=11)

                sub_ax2.tick_params(axis="x", labelsize=16)
                sub_ax2.tick_params(axis="y", labelsize=11)

                sub_ax0.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False,
                )
                sub_ax1.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False,
                )

                if target_indices.index(target_index) in [0, 1]:
                    sub_ax2.tick_params(
                        axis="x",
                        which="both",
                        bottom=True,
                        top=False,
                        labelbottom=False,
                    )

                if target_indices.index(target_index) in [1, 3]:
                    sub_ax0.tick_params(
                        axis="y",
                        which="both",
                        labelleft=False,
                        left=True,
                    )
                    sub_ax1.tick_params(
                        axis="y",
                        which="both",
                        labelleft=False,
                        left=True,
                    )
                    sub_ax2.tick_params(
                        axis="y",
                        which="both",
                        labelleft=False,
                        left=True,
                    )

            fig.text(0.5, -0.01, "Time Step", ha="center", fontsize=16)
            fig.text(0.07, 0.5, "Value", va="center", rotation="vertical", fontsize=16)

            handles, labels = sub_ax0.get_legend_handles_labels()

            handles.append(mpatches.Patch(color="red", label="Positive Attribution"))
            labels.append("Positive Attribution")
            handles.append(mpatches.Patch(color="blue", label="Negative Attribution"))
            labels.append("Negative Attribution")

            fig.savefig(
                exp_path / "results_pdf" / f"{exp_attr}_{output}.pdf",
                bbox_inches="tight",
            )
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                bbox_to_anchor=(0.5, 1.1),
                facecolor="white",
                fontsize=16,
            )

            plt.show()

            fig.savefig(
                exp_path / "results_pdf" / f"{exp_attr}_{output}_legend.pdf",
                bbox_inches="tight",
            )

fig, axes = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(12, 6))
fig.subplots_adjust(wspace=0.08, hspace=0.3)

flattened_axes = axes.flatten()
flattened_axes[-1].axis("off")
for i, (ax, exp_model) in enumerate(
    zip(flattened_axes[:3], ["mlp", "nbeats", "nhits"])
):
    exp_root = Path("../experimental_runs")
    exp_path = exp_root / EXP_NAME / exp_model / "results_csv"
    predictions_path = exp_path / "predictions.csv"
    fa_path = exp_path / "FA"

    predictions_df = pd.read_csv(predictions_path)

    # get top bottom right and left of ax
    top = ax.get_position().get_points()[1][1]
    bottom = ax.get_position().get_points()[0][1]
    right = ax.get_position().get_points()[1][0]
    left = ax.get_position().get_points()[0][0]
    ax.axis("off")

    gs = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        top=top,
        bottom=bottom,
        right=right,
        left=left,
        hspace=0.1,
    )
    sub_ax0 = fig.add_subplot(gs[0])
    sub_ax1 = fig.add_subplot(gs[1])
    sub_ax2 = fig.add_subplot(gs[2])

    csv_path = fa_path / "loc_attributions_0.csv"
    df = pd.read_csv(csv_path)

    input_x = np.arange(len(df["ds"]))
    prediction_x = np.arange(len(df["ds"]), len(df["ds"]) + len(predictions_df))

    y = df["y"]
    sine_a = df["sine_spiky_a"]
    sine_b = df["sine_spiky_b"]

    sub_ax0.plot(input_x, y, color="black", label="Input")
    sub_ax1.plot(input_x, sine_a, color="black", label="Input")
    sub_ax2.plot(input_x, sine_b, color="black", label="Input")

    # Plot the predictions
    sub_ax0.plot(
        prediction_x,
        predictions_df[f"{model_to_output[exp_model]}-loc"],
        label=f"Mean Prediction",
        color="green",
    )

    # Fill between for the variance
    sub_ax0.fill_between(
        prediction_x,
        predictions_df[f"{model_to_output[exp_model]}-loc"]
        - predictions_df[f"{model_to_output[exp_model]}-scale"],
        predictions_df[f"{model_to_output[exp_model]}-loc"]
        + predictions_df[f"{model_to_output[exp_model]}-scale"],
        label=f"Std. Dev. Prediction",
        color="green",
        alpha=0.2,
    )

    sub_ax0.set_ylim(-5, 5)
    sub_ax1.set_ylim(-5, 5)
    sub_ax2.set_ylim(-5, 5)

    sub_ax0.set_yticks([-3, 0, 3])
    sub_ax1.set_yticks([-3, 0, 3])
    sub_ax2.set_yticks([-3, 0, 3])

    sub_ax0.set_xlim(0, 144)
    sub_ax1.set_xlim(0, 144)
    sub_ax2.set_xlim(0, 144)

    sub_ax0.tick_params(axis="x", labelsize=16)
    sub_ax0.tick_params(axis="y", labelsize=11)

    sub_ax1.tick_params(axis="x", labelsize=16)
    sub_ax1.tick_params(axis="y", labelsize=11)

    sub_ax2.tick_params(axis="x", labelsize=16)
    sub_ax2.tick_params(axis="y", labelsize=11)

    sub_ax0.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )
    sub_ax1.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    if i in [0]:
        sub_ax2.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
        )

    if i in [1, 3]:
        sub_ax0.tick_params(
            axis="y",
            which="both",
            labelleft=False,
            left=True,
        )
        sub_ax1.tick_params(
            axis="y",
            which="both",
            labelleft=False,
            left=True,
        )
        sub_ax2.tick_params(
            axis="y",
            which="both",
            labelleft=False,
            left=True,
        )

    sub_ax0.set_title(f"{model_to_display[exp_model]}", fontsize=16)

fig.text(0.5, -0.01, "Time Step", ha="center", fontsize=16)
fig.text(0.07, 0.5, "Value", va="center", rotation="vertical", fontsize=16)

handles, labels = sub_ax0.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.03),
    facecolor="white",
    fontsize=16,
)
plt.savefig(exp_root / EXP_NAME / "all_models_top.pdf", bbox_inches="tight")
plt.show()
