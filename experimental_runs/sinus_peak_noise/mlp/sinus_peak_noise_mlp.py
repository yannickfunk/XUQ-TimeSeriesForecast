from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import MLP

from experiment_templates.sinus_peak_noise import (HORIZON, INPUT_SIZE, LEVELS,
                                                   run)
from nf_ti_adapter.nhits import NhitsNfTiAdapter

model = MLP(
    input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=1500,
    random_seed=40,
    # early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
    scaler_type="robust",
)
adapter = NhitsNfTiAdapter(model, 1)

run(model, adapter)
