from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import MLP

from experiment_templates.exogenous_sinus_peak_spike import (HORIZON,
                                                             INPUT_SIZE,
                                                             LEVELS, run)
from nf_ti_adapter.nhits import NhitsNfTiAdapter

model = MLP(
    input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=2500,
    random_seed=40,
    # early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
    hist_exog_list=["sine_spiky_a", "sine_spiky_b"],
    scaler_type="robust",
    # exclude_insample_y=True,
)

adapter = NhitsNfTiAdapter(model, 1)

run(model, adapter)
