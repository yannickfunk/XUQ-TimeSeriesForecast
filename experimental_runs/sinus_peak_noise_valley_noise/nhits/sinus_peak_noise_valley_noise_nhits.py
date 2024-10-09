from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NHITS

from experiment_templates.sinus_peak_noise_valley_noise import (HORIZON,
                                                                INPUT_SIZE,
                                                                LEVELS, run)
from nf_ti_adapter.nhits import NhitsNfTiAdapter

model = NHITS(
    input_size=INPUT_SIZE,
    # inference_input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=600,
    random_seed=4567,
    # early_stop_patience_steps=5,
    # val_check_steps=10,
    scaler_type="robust",
    logger=TensorBoardLogger("logs"),
)

adapter = NhitsNfTiAdapter(model, 1)

run(model, adapter)
