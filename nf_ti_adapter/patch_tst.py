from typing import Union

import pandas as pd
import torch

from nf_ti_adapter.base import Model, NfTiAdapter


class PatchTstNfTiAdapter(NfTiAdapter):
    def __init__(self, model: Model, freq: Union[str, int]):
        if str(model) != "PatchTST":
            raise ValueError("Model must be an instance of PatchTST")
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

        if "-lo-" in output_name:
            return (
                model_output[0, :, output_index + 1] - model_output[0, :, output_index]
            )

        return model_output[0, :, output_index]
