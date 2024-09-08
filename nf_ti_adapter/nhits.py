from typing import Union

import pandas as pd
import torch

from nf_ti_adapter.base import Model, NfTiAdapter


class NhitsNfTiAdapter(NfTiAdapter):
    def __init__(self, model: Model, freq: Union[str, int]):
        if str(model) not in ["NHITS", "NBEATS", "NBEATSx", "MLP"]:
            raise ValueError("Model must be an instance of NHITS")
        super().__init__(model, freq)

    def _forward_function(
        self, inputs: torch.Tensor, output_name: str, output_uid: Union[str, int] = None
    ):
        output_index = self.output_names.index(output_name)
        if output_uid is not None:
            uid_index = list(self.nf.uids.values).index(output_uid)
        else:
            uid_index = 0
        # check multiple or single time series inputs
        if len(self.nf.uids) > 1:
            inputs = inputs[0].T
            masks = torch.ones_like(inputs)
            inputs = torch.stack([inputs, masks], dim=1)
        elif len(self.nf.dataset.temporal_cols) > 2:
            masks = torch.ones((inputs.shape[0], inputs.shape[1], 1))
            inputs = torch.cat((inputs, masks), dim=2)
            inputs = inputs.permute(0, 2, 1)
        else:
            inputs = inputs[0, ..., 0]
            masks = torch.ones_like(inputs)
            inputs = torch.unsqueeze(torch.vstack([inputs, masks]), 0)
        batch = {
            "temporal": inputs,
            "temporal_cols": self.nf.dataset.temporal_cols,
            "y_idx": 0,
        }
        batch_idx = 0
        model_output = self.model.predict_step(batch, batch_idx)

        if "-lo-" in output_name:
            return (
                model_output[uid_index, :, output_index + 1]
                - model_output[uid_index, :, output_index]
            )

        return model_output[uid_index, :, output_index]
