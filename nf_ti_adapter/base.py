from typing import TypeVar, Union

from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import LSTM

Model = TypeVar("Model", bound=BaseModel)


class NfTiAdapter:
    def __init__(self, model: Model, freq: Union[str, int]):
        self.model = model
        self.freq = freq

        self.nf = NeuralForecast(
            models=[model],
            freq=freq,
        )

        self.horizon = self.model.h
        self.output_names = self.model.loss.output_names


# Test
if __name__ == "__main__":
    HORIZON = 86
    LEVELS = [80, 90]
    INPUT_SIZE = 3 * HORIZON

    model = LSTM(
        input_size=INPUT_SIZE,
        h=HORIZON,
        loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
        max_steps=500,
    )

    explainer = NfTiAdapter(model, 1)
