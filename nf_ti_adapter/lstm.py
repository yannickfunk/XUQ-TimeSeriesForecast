from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tint.attr import AugmentedOcclusion, TemporalIntegratedGradients

from nf_ti_adapter.base import Model, NfTiAdapter


class LstmNfTiAdapter(NfTiAdapter):
    def __init__(self, model: Model, freq: Union[str, int]):
        if str(model) != "LSTM":
            raise ValueError("Model must be an instance of LSTM")
        super().__init__(model, freq)

    def _forward_function(self, inputs: torch.Tensor, output_name: str):
        output_index = self.output_names.index(output_name)
        inputs = inputs[0, :, 0]
        masks = torch.ones_like(inputs)
        inputs = torch.unsqueeze(torch.vstack([inputs, masks]), 0)
        batch = {
            "temporal": inputs,
            "temporal_cols": pd.Index(["y", "available_mask"]),
            "y_idx": 0,
        }
        batch_idx = 0
        model_output = self.model.predict_step(batch, batch_idx)
        return model_output[0, -1, :, output_index]

    def explain(
        self,
        method: str,
        target_indices: List[int],
        ds: Optional[Union[List[str], np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
    ):
        if not hasattr(self.nf, "ds"):
            raise ValueError("Model has to be trained before calling explain")
        self._sanity_check()
        if ds is None and y is None:
            ds, y = self._get_current_inference_data()
        elif ds is not None and y is not None:
            pass
        else:
            raise ValueError("ds and y both have to be None, or both not None")

        # add batch and feature dimension
        y = y[-self.model.inference_input_size :]
        y = torch.unsqueeze(y, 0)
        y = torch.unsqueeze(y, -1)

        method_to_constructor = {
            "TIG": TemporalIntegratedGradients,
            "AugOcc": AugmentedOcclusion,
        }
        attributions = []
        for target_idx in target_indices:
            forward_callable = lambda x: self._forward_function(x, "-scale")[
                target_idx : target_idx + 1
            ]

            explanation_method: TemporalIntegratedGradients = method_to_constructor[
                method
            ](forward_callable)
            attr = explanation_method.attribute(y, show_progress=True).abs()
            attributions.append(attr)

        return attributions
