from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel  # noqa

Model = TypeVar("Model", bound=BaseModel)


class NfTiAdapter:
    def __init__(self, model: Model, freq: Union[str, int]):
        self.model = model
        self.freq = freq

        self.nf = NeuralForecast(
            models=[model],
            freq=freq,
        )

        self.horizon: int = self.model.h
        self.output_names: List[str] = self.model.loss.output_names
        (self.point_output, self.levels, self.parametric_output) = (
            self._parse_output_names()
        )

    def _parse_output_names(self) -> Tuple[bool, List[int], bool]:
        levels = []
        parametric_output = False
        point_output = False
        for name in self.output_names:
            if name == "":
                point_output = True
            if "-lo-" in name:
                levels.append(int(name.split("-")[-1]))
            if "loc" in name:
                parametric_output = True
        return point_output, sorted(levels), parametric_output

    @staticmethod
    def _create_nf_dataframe(ds: Union[List[str], np.ndarray], y: np.ndarray):
        return pd.DataFrame(
            {
                "unique_id": 1.0,
                "ds": ds,
                "y": y,
            }
        )

    def _get_current_train_data(self):
        return (
            self.nf.ds,
            self.nf.dataset.temporal[:, self.nf.dataset.temporal_cols.get_loc("y")],
        )

    def _forward_function(self, inputs: torch.Tensor, output_name: str):
        raise NotImplementedError

    def _sanity_check(self):
        _, train_y = self._get_current_train_data()
        # add batch dimension
        train_y = torch.unsqueeze(train_y, 0)

        # add feature dimension
        train_y = torch.unsqueeze(train_y, -1)

        for output_name in self.output_names:
            predictions = self.nf.predict()[f"{self.model}{output_name}"]
            raw_predictions = self._forward_function(train_y, output_name)
            match = np.allclose(
                predictions, raw_predictions.detach().numpy(), rtol=1e-1, atol=1e-1
            )
            if not match:
                raise ValueError(
                    f"Model predictions do not match for output {output_name}"
                )

    def fit(self, ds: Union[List[str], np.ndarray], y: np.ndarray):
        self.nf.fit(df=self._create_nf_dataframe(ds, y))

        # update model
        self.model = self.nf.models[0]

    def predict(
        self,
        ds: Optional[Union[List[str], np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
    ):
        if ds is None and y is None:
            return self.nf.predict()
        elif ds is not None and y is not None:
            return self.nf.predict(df=self._create_nf_dataframe(ds, y))
        else:
            raise ValueError("ds and y both have to be None, or both not None")

    def predict_plot(
        self,
        ds: Optional[Union[List[str], np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        test_ds: Optional[Union[List[str], np.ndarray]] = None,
        test_y: Optional[np.ndarray] = None,
    ):
        predictions = self.predict(ds, y)

        if self.point_output:
            self._plot_predictions_point(predictions, test_ds, test_y)
        if len(self.levels) > 0:
            self._plot_predictions_quantile(predictions, test_ds, test_y)
        if self.parametric_output:
            self._plot_predictions_parametric(predictions, test_ds, test_y)

    def _plot_predictions_point(
        self,
        predictions: pd.DataFrame,
        test_ds: Optional[Union[List[str], np.ndarray]],
        test_y: Optional[np.ndarray],
    ):
        last = self.horizon
        train_ds, train_y = self._get_current_train_data()

        plt.plot(
            train_ds[-last:],
            train_y[-last:],
            label="train data",
        )
        if test_ds is not None and test_y is not None:
            plt.plot(test_ds, test_y, label="test data", color="red")
        plt.plot(
            predictions.ds,
            predictions[f"{self.model}"],
            label="prediction",
            color="green",
        )
        plt.title(f"Predictions, model: {self.model}")
        plt.legend(loc="upper left")
        plt.show()

    def _plot_predictions_quantile(
        self,
        predictions: pd.DataFrame,
        test_ds: Optional[Union[List[str], np.ndarray]],
        test_y: Optional[np.ndarray],
    ):
        last = self.horizon
        train_ds, train_y = self._get_current_train_data()

        plt.plot(
            train_ds[-last:],
            train_y[-last:],
            label="train data",
        )
        if test_ds is not None and test_y is not None:
            plt.plot(test_ds, test_y, label="test data", color="red")
        plt.plot(
            predictions.ds,
            predictions[f"{self.model}-median"],
            label="median prediction",
            color="green",
        )

        for i, level in enumerate(self.levels):
            plt.fill_between(
                predictions.ds,
                predictions[f"{self.model}-lo-{level}"],
                predictions[f"{self.model}-hi-{level}"],
                alpha=1 - (i + 1) * (1 / (len(self.levels) + 1)),
                color="orange",
                label=f"{level}% prediction interval",
            )
        plt.title(f"Predictions with prediction intervals, model: {self.model}")
        plt.legend(loc="upper left")
        plt.show()

    def _plot_predictions_parametric(
        self,
        predictions: pd.DataFrame,
        test_ds: Optional[Union[List[str], np.ndarray]],
        test_y: Optional[np.ndarray],
    ):
        last = self.horizon
        train_ds, train_y = self._get_current_train_data()

        plt.plot(
            train_ds[-last:],
            train_y[-last:],
            label="train data",
        )
        if test_ds is not None and test_y is not None:
            plt.plot(test_ds, test_y, label="test data", color="red")

        plt.plot(
            predictions.ds,
            predictions[f"{self.model}-loc"],
            label="mean prediction",
            color="green",
        )
        plt.fill_between(
            predictions.ds,
            np.subtract(
                predictions[f"{self.model}-loc"], predictions[f"{self.model}-scale"]
            ),
            np.add(
                predictions[f"{self.model}-loc"], predictions[f"{self.model}-scale"]
            ),
            alpha=0.2,
            color="green",
            label="standard deviation",
        )

        plt.title(f"Mean prediction and standard deviation, model: {self.model}")
        plt.legend(loc="upper left")
        plt.show()
