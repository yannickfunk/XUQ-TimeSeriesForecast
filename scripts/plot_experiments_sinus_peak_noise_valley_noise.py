from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")

EXP_NAME = "sinus_peak_noise_valley_noise"
exp_root = Path("../experimental_runs")

model_to_output = {
    "mlp": "MLP",
    "nbeats": "NBEATS",
    "nhits": "NHITS",
    "patch_tst": "PatchTST",
}

model_to_display = {
    "mlp": "MLP",
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "patch_tst": "PatchTST",
}

for exp_model in ["mlp", "nbeats", "nhits", "patch_tst"]:
    exp_path = exp_root / EXP_NAME / exp_model
    csv_results_path = exp_path / "results_csv"
    predictions_path = csv_results_path / "predictions.csv"

    for exp_attr in ["FA", "IG", "IXG", "LIME"]:
        for output in ["loc", "scale"]:
            fa_path = csv_results_path / exp_attr
            predictions_df = pd.read_csv(predictions_path)

            fig, axes = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(12, 5))
            target_indices = [18, 28, 33, 38]

            for ax, target_index in zip(axes.flatten(), target_indices):
                csv_path = fa_path / f"{output}_attributions_{target_index}.csv"
                df = pd.read_csv(csv_path)

                pos_attr = df["pos_attr"]
                neg_attr = df["neg_attr"]
                # raw_pos_attr = df["raw_pos_attr"]
                # raw_neg_attr = df["raw_neg_attr"]
                # raw_attr = raw_pos_attr + raw_neg_attr

                input_x = np.arange(len(df["ds"]))
                prediction_x = np.arange(
                    len(df["ds"]), len(df["ds"]) + len(predictions_df)
                )

                # Plot the positive and negative attributions as vertical lines
                for i in range(len(pos_attr)):
                    ax.axvline(x=input_x[i], color="red", alpha=pos_attr.iloc[i], lw=3)
                    ax.axvline(x=input_x[i], color="blue", alpha=neg_attr.iloc[i], lw=3)

                # Plot the true values
                ax.plot(input_x, df["y"], label=f"Input", color="black")

                # Plot the predictions
                ax.plot(
                    prediction_x,
                    predictions_df[f"{model_to_output[exp_model]}-loc"],
                    label=f"Mean Prediction",
                    color="green",
                )

                # Fill between for the variance
                ax.fill_between(
                    prediction_x,
                    predictions_df[f"{model_to_output[exp_model]}-loc"]
                    - predictions_df[f"{model_to_output[exp_model]}-scale"],
                    predictions_df[f"{model_to_output[exp_model]}-loc"]
                    + predictions_df[f"{model_to_output[exp_model]}-scale"],
                    label=f"Std. Dev. Prediction",
                    color="green",
                    alpha=0.2,
                )

                ax.plot(
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

                # # plot the raw attributions
                # ax.plot(input_x, raw_attr * 1000, color="purple")

                ax.tick_params(axis="x", labelsize=16)
                ax.tick_params(axis="y", labelsize=16)

            fig.subplots_adjust(wspace=0.05, hspace=0.1)

            handles, labels = plt.gca().get_legend_handles_labels()

            handles.append(mpatches.Patch(color="red", label="Positive Attribution"))
            labels.append("Positive Attribution")
            handles.append(mpatches.Patch(color="blue", label="Negative Attribution"))
            labels.append("Negative Attribution")

            fig.text(0.5, 0.0, "Time Step", ha="center", fontsize=16)
            fig.text(0.07, 0.5, "Value", va="center", rotation="vertical", fontsize=16)
            plt.xlim(0, 144)
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

fig, axes = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(12, 5))
for ax, exp_model in zip(axes.flatten(), ["mlp", "nbeats", "nhits", "patch_tst"]):
    exp_root = Path("../experimental_runs")
    exp_path = exp_root / EXP_NAME / exp_model / "results_csv"
    predictions_path = exp_path / "predictions.csv"
    fa_path = exp_path / "FA"

    predictions_df = pd.read_csv(predictions_path)

    csv_path = fa_path / f"loc_attributions_{0}.csv"
    df = pd.read_csv(csv_path)

    pos_attr = df["pos_attr"]
    neg_attr = df["neg_attr"]
    input_x = np.arange(len(df["ds"]))
    prediction_x = np.arange(len(df["ds"]), len(df["ds"]) + len(predictions_df))

    # Plot the true values
    ax.plot(input_x, df["y"], label=f"Input", color="black")

    # Plot the predictions
    ax.plot(
        prediction_x,
        predictions_df[f"{model_to_output[exp_model]}-loc"],
        label=f"Mean Prediction",
        color="green",
    )

    # Fill between for the variance
    ax.fill_between(
        prediction_x,
        predictions_df[f"{model_to_output[exp_model]}-loc"]
        - predictions_df[f"{model_to_output[exp_model]}-scale"],
        predictions_df[f"{model_to_output[exp_model]}-loc"]
        + predictions_df[f"{model_to_output[exp_model]}-scale"],
        label=f"Std. Dev. Prediction",
        color="green",
        alpha=0.2,
    )
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_title(f"{model_to_display[exp_model]}", fontsize=16)

fig.subplots_adjust(wspace=0.05, hspace=0.35)

handles, labels = plt.gca().get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.14),
    facecolor="white",
    fontsize=16,
)

fig.text(0.5, 0.0, "Time Step", ha="center", fontsize=16)
fig.text(0.07, 0.5, "Value", va="center", rotation="vertical", fontsize=16)
plt.xlim(0, 144)
plt.savefig(exp_root / EXP_NAME / "all_models.pdf", bbox_inches="tight")
plt.show()
