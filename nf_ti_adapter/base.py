from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import tikzplotlib
import torch
from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel  # noqa
from statsforecast import StatsForecast
from tint.attr import AugmentedOcclusion, TemporalIntegratedGradients

from common.timeseries import AttributedTimeSeries, TimeSeries

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
        self.inference_input_size = getattr(
            self.model, "inference_input_size", self.model.input_size
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
    def _create_nf_dataframe(
        ds: Union[List[str], np.ndarray], y: np.ndarray, unique_id: Union[int, str] = 1
    ):
        return pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": ds,
                "y": y,
            }
        )

    @staticmethod
    def _create_nf_dataframe_list(time_series_list: List[TimeSeries]):
        return pd.concat(
            [
                NfTiAdapter._create_nf_dataframe(ts.ds, ts.y, unique_id=ts.unique_id)
                for ts in time_series_list
            ]
        )

    @staticmethod
    def _create_nf_dataframe_list_exogenous(
        time_series_list: List[TimeSeries], target_uid
    ):
        fit_df_dict = {}
        for ts in time_series_list:
            if ts.unique_id == target_uid:
                fit_df_dict["unique_id"] = ts.unique_id
                fit_df_dict["y"] = ts.y
                fit_df_dict["ds"] = ts.ds
            else:
                fit_df_dict[ts.unique_id] = ts.y
        fit_df = pd.DataFrame(fit_df_dict)
        return fit_df

    def _get_current_train_data(self):
        temporal_data = self.nf.dataset.temporal[
            :, self.nf.dataset.temporal_cols.get_loc("y")
        ]
        if len(self.nf.uids) == 1:
            return self.nf.ds, temporal_data

        temporal_stacked = torch.zeros((len(self.nf.uids), self.nf.dataset.indptr[1]))
        for i in range(len(self.nf.uids)):
            temporal_stacked[i] = temporal_data[
                self.nf.dataset.indptr[i] : self.nf.dataset.indptr[i + 1]
            ]
        return self.nf.ds, temporal_stacked.T

    def _forward_function(
        self, inputs: torch.Tensor, output_name: str, output_uid: Union[str, int] = None
    ):
        raise NotImplementedError

    def _sanity_check(self):
        """
        train_y = self._get_current_train_data()[1]

        # add batch dimension
        train_y = torch.unsqueeze(train_y, 0)

        # add feature dimension
        if len(self.nf.uids) == 1:
            train_y = torch.unsqueeze(train_y, -1)

        for output_name in self.output_names:
            predictions = self.nf.predict()[f"{self.model}{output_name}"]
            raw_predictions = self._forward_function(train_y, output_name)
            plt.plot(predictions, label="predictions")
            plt.plot(raw_predictions.detach().numpy(), label="raw predictions")
            plt.title(f"Model predictions for output {output_name}")
            plt.legend()
            plt.show()
            match = np.allclose(
                predictions, raw_predictions.detach().numpy(), rtol=1e-1, atol=1e-1
            )
            if not match:
                raise ValueError(
                    f"Model predictions do not match for output {output_name}"
                )
        """
        pass

    def _sanity_check_exog(self, time_series_list):
        """
        ds, y, uid_exog_list = self._arrays_from_time_series_list(time_series_list)

        # add batch dimension
        y = y[-self.inference_input_size :]
        y = torch.unsqueeze(y, 0)
        for output_name in self.output_names:
            raw_predictions = self._forward_function(y, output_name)
            break
        """
        pass

    def fit(self, ds: Union[List[str], np.ndarray], y: np.ndarray, **kwargs):
        self.nf.fit(df=self._create_nf_dataframe(ds, y), **kwargs)

        # update model
        self.model = self.nf.models[0]

    def fit_list(self, time_series_list: List[TimeSeries], **kwargs):
        fit_df = self._create_nf_dataframe_list(time_series_list)
        self.nf.fit(df=fit_df, **kwargs)

        # update model
        self.model = self.nf.models[0]

    def fit_list_exogenous(
        self, time_series_list: List[TimeSeries], target_uid, **kwargs
    ):
        fit_df = self._create_nf_dataframe_list_exogenous(time_series_list, target_uid)
        self.nf.fit(df=fit_df, **kwargs)

        # update model
        self.model = self.nf.models[0]

    def predict(
        self,
        ds: Optional[Union[List[str], np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if ds is None and y is None:
            return self.nf.predict()
        elif ds is not None and y is not None:
            return self.nf.predict(df=self._create_nf_dataframe(ds, y))
        else:
            raise ValueError("ds and y both have to be None, or both not None")

    def predict_list(
        self,
        time_series_list: List[TimeSeries],
    ) -> pd.DataFrame:
        return self.nf.predict(df=self._create_nf_dataframe_list(time_series_list))

    def predict_list_plot(
        self,
        time_series_list: List[TimeSeries],
    ) -> pd.DataFrame:
        test_input = self._create_nf_dataframe_list(time_series_list)
        predictions = self.nf.predict(df=test_input)
        fig = StatsForecast.plot(
            test_input,
            predictions,
            level=self.levels,
            models=[str(self.model)],
        )
        fig.show()

        return predictions

    def predict_list_exogenous_plot(
        self,
        time_series_list: List[TimeSeries],
        target_uid: str,
    ) -> pd.DataFrame:
        test_input = self._create_nf_dataframe_list_exogenous(
            time_series_list, target_uid=target_uid
        )
        predictions = self.nf.predict(df=test_input)

        # plot time series list with train / test split borders as vertical lines
        fig, axs = plt.subplots(len(time_series_list), 1, sharex="all")
        for i, ts in enumerate(time_series_list):
            if ts.unique_id == target_uid:
                axs[i].plot(test_input.ds, test_input["y"])
                axs[i].plot(
                    predictions.ds,
                    predictions[f"{self.model}"],
                    label="predictions",
                    color="green",
                )
                axs[i].fill_between(
                    predictions.ds,
                    np.subtract(
                        predictions[f"{self.model}-loc"],
                        predictions[f"{self.model}-scale"],
                    ),
                    np.add(
                        predictions[f"{self.model}-loc"],
                        predictions[f"{self.model}-scale"],
                    ),
                    alpha=0.2,
                    color="green",
                    label="standard deviation",
                )
                axs[i].set_title(f"{ts.unique_id} - predictions")
            else:
                axs[i].plot(test_input.ds, test_input[ts.unique_id])
                axs[i].set_title(ts.unique_id)

            axs[i].set_ylim(-6, 6)
            axs[i].legend(loc="upper left")
            fig.subplots_adjust(hspace=0.6)
        plt.show()
        return predictions

    def predict_plot(
        self,
        ds: Union[List[str], np.ndarray],
        y: np.ndarray,
        test_ds: Optional[Union[List[str], np.ndarray]] = None,
        test_y: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        predictions = self.predict(ds, y)

        if self.point_output:
            self._plot_predictions_point(ds, y, predictions, test_ds, test_y)
        if len(self.levels) > 0:
            self._plot_predictions_quantile(ds, y, predictions, test_ds, test_y)
        if self.parametric_output:
            self._plot_predictions_parametric(ds, y, predictions, test_ds, test_y)
        return predictions

    def _plot_predictions_point(
        self,
        ds: Union[List[str], np.ndarray],
        y: np.ndarray,
        predictions: pd.DataFrame,
        test_ds: Optional[Union[List[str], np.ndarray]],
        test_y: Optional[np.ndarray],
    ):
        if test_ds is not None and test_y is not None:
            # get test_y where test_ds equals predictions.ds
            test_y = test_y[np.isin(test_ds, predictions.ds)]
            plt.plot(predictions.ds, test_y, label="test data", color="red")
        plt.plot(
            predictions.ds,
            predictions[f"{self.model}"],
            label="prediction",
            color="green",
        )
        plt.plot(
            ds,
            y,
            label="input data",
        )
        plt.title(f"Predictions, model: {self.model}")
        plt.legend(loc="upper left")
        plt.savefig(f"results/predictions_point.png")
        tikzplotlib.save("results_tikz/predictions_point.tex")
        plt.show()

    def _plot_predictions_quantile(
        self,
        ds: Union[List[str], np.ndarray],
        y: np.ndarray,
        predictions: pd.DataFrame,
        test_ds: Optional[Union[List[str], np.ndarray]],
        test_y: Optional[np.ndarray],
    ):
        if test_ds is not None and test_y is not None:
            # get test_y where test_ds equals predictions.ds
            test_y = test_y[np.isin(test_ds, predictions.ds)]
            plt.plot(predictions.ds, test_y, label="test data", color="red")

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
        plt.plot(
            ds,
            y,
            label="input data",
        )
        plt.title(f"Predictions with prediction intervals, model: {self.model}")
        plt.legend(loc="upper left")
        plt.savefig(f"results/predictions_quantile.png")
        tikzplotlib.save("results_tikz/predictions_quantile.tex")
        plt.show()

    def _plot_predictions_parametric(
        self,
        ds: Union[List[str], np.ndarray],
        y: np.ndarray,
        predictions: pd.DataFrame,
        test_ds: Optional[Union[List[str], np.ndarray]],
        test_y: Optional[np.ndarray],
    ):
        if test_ds is not None and test_y is not None:
            # get test_y where test_ds equals predictions.ds
            test_y = test_y[np.isin(test_ds, predictions.ds)]
            plt.plot(predictions.ds, test_y, label="test data", color="red")

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
        plt.plot(
            ds,
            y,
            label="input data",
        )

        plt.title(f"Mean prediction and standard deviation, model: {self.model}")
        plt.legend(loc="upper left")
        plt.savefig(f"results/predictions_parametric.png")
        tikzplotlib.save("results_tikz/predictions_parametric.tex")
        plt.show()

    def explain(
        self,
        method: str,
        target_indices: List[int],
        output_name: str,
        ds: Union[List[str], np.ndarray],
        y: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not hasattr(self.nf, "ds"):
            raise ValueError("Model has to be trained before calling explain")
        if output_name not in self.output_names:
            raise ValueError(f"output_name {output_name} not in {self.output_names}")
        self._sanity_check()

        # add batch and feature dimension
        y = torch.tensor(y, dtype=torch.float32)
        y = y[-self.inference_input_size :]
        y = torch.unsqueeze(y, 0)
        y = torch.unsqueeze(y, -1)

        method_to_constructor = {
            "TIG": TemporalIntegratedGradients,
            "AugOcc": AugmentedOcclusion,
        }
        attributions = []
        negative_attributions = []
        for target_idx in target_indices:
            forward_callable = lambda x: self._forward_function(x, output_name)[
                target_idx : target_idx + 1
            ]

            explanation_method: TemporalIntegratedGradients = method_to_constructor[
                method
            ](forward_callable)
            attr = explanation_method.attribute(y, show_progress=True)[0, ...]
            attr = torch.nan_to_num(attr)
            attr = attr.detach().numpy().squeeze()
            max_attr = np.max(np.abs(attr))

            negative_attr = np.abs(attr.clip(max=0)) / max_attr
            negative_attributions.append(negative_attr)

            positive_attr = np.abs(attr.clip(min=0)) / max_attr
            attributions.append(positive_attr)

        return attributions, negative_attributions

    def explain_list(
        self,
        method: str,
        target_indices: List[int],
        output_name: str,
        output_uid: str,
        test_input_list: List[TimeSeries],
    ) -> List[AttributedTimeSeries]:
        if not hasattr(self.nf, "ds"):
            raise ValueError("Model has to be trained before calling explain")
        if output_name not in self.output_names:
            raise ValueError(f"output_name {output_name} not in {self.output_names}")
        self._sanity_check_exog(test_input_list)
        ds, y, uid_exog_list = self._arrays_from_time_series_list(test_input_list)

        # add batch dimension
        y = y[-self.inference_input_size :]
        y = torch.unsqueeze(y, 0)

        method_to_constructor = {
            "TIG": TemporalIntegratedGradients,
            "AugOcc": AugmentedOcclusion,
        }
        attributions = []
        negative_attributions = []
        for target_idx in target_indices:
            forward_callable = lambda x: self._forward_function(
                x, output_name, output_uid
            )[target_idx : target_idx + 1]

            explanation_method: TemporalIntegratedGradients = method_to_constructor[
                method
            ](forward_callable)
            attr = explanation_method.attribute(y, show_progress=True)[0, ...]
            attr = torch.nan_to_num(attr)
            attr = attr.detach().numpy().squeeze()
            max_attr = np.max(np.abs(attr))

            negative_attr = np.abs(attr.clip(max=0)) / max_attr
            negative_attributions.append(negative_attr)

            positive_attr = np.abs(attr.clip(min=0)) / max_attr
            attributions.append(positive_attr)

        attributed_timeseries_list = []
        for idx, uid in enumerate(uid_exog_list):
            _y = y[0, :, idx].detach().numpy()
            _positive_attributions = [attr[:, idx] for attr in attributions]
            _negative_attributions = [attr[:, idx] for attr in negative_attributions]
            attributed_timeseries_list.append(
                AttributedTimeSeries(
                    unique_id=uid,
                    ds=ds,
                    y=_y,
                    positive_attributions=_positive_attributions,
                    negative_attributions=_negative_attributions,
                )
            )

        return attributed_timeseries_list

    def _arrays_from_time_series_list(self, time_series_list: List[TimeSeries]):
        ds = time_series_list[0].ds

        # sort time_series_list by unique_id
        uid_map = {uid: idx for idx, uid in enumerate(self.nf.uids)}
        exog_list = list(self.nf.dataset.temporal_cols)[1:-1]
        for i, exog in enumerate(exog_list):
            uid_map[exog] = len(self.nf.uids) + i

        # get uid_exog_list from the entries in uid_map
        uid_exog_list = [
            k for k, v in sorted(uid_map.items(), key=lambda item: item[1])
        ]

        sorted_time_series_list = sorted(
            time_series_list, key=lambda x: uid_map[x.unique_id]
        )
        y = np.vstack([ts.y for ts in sorted_time_series_list])

        return ds, torch.tensor(y.T, dtype=torch.float32), uid_exog_list
