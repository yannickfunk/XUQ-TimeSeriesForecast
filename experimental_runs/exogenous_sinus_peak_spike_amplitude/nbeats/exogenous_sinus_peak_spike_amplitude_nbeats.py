from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NBEATSx

from experiment_templates.exogenous_sinus_peak_spike_amplitude import (
    HORIZON, INPUT_SIZE, run)
from nf_ti_adapter.nhits import NhitsNfTiAdapter

model = NBEATSx(
    input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=[80, 90], return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=600,
    random_seed=40,
    # early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
    hist_exog_list=["sine_spiky_a", "sine_spiky_b"],
    scaler_type="robust",
    # exclude_insample_y=True,
)

adapter = NhitsNfTiAdapter(model, 1)

run(model, adapter)
