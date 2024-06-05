from typing import Union

import pandas as pd
import torch

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
