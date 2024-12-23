from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = [16, 6]
import pandas as pd
from lightning.pytorch.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NHITS
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from common.timeseries import TimeSeries
from common.utils import plot_attributions
from nf_ti_adapter.base import METHOD_TO_CONSTRUCTOR
from nf_ti_adapter.nhits import NhitsNfTiAdapter

plt.style.use("ggplot")

DATA_ROOT = Path("~/data/entsoe").expanduser()
COUNTRY = "FR"
START_YEAR = 2015
END_YEAR = 2023
FEATURE = "load"


def load_df():
    df = pd.DataFrame()

    # read csv files from 2019 to 2023 and concatenate them
    for year in range(START_YEAR, END_YEAR + 1):
        csv_dir = DATA_ROOT / COUNTRY / str(year)
        csv_file_path = f"{FEATURE}_{COUNTRY}_{year}.csv"
        year_df = pd.read_csv(csv_dir / csv_file_path)
        df = pd.concat([df, year_df])
    return df


def preprocess_df(df):
    prep_df = df.rename(columns={df.columns[0]: "ds"})

    if FEATURE != "generation":
        prep_df = prep_df.rename(columns={prep_df.columns[1]: FEATURE})
    prep_df["ds"] = pd.to_datetime(prep_df["ds"]).dt.tz_localize(None)
    prep_df = prep_df.rename(columns={FEATURE: "y"})
    prep_df = prep_df.sort_values("ds")

    # replace nans in prep_df_2023.y with next value
    prep_df.y = prep_df.y.ffill()

    # scale prep_df.y with sklearn robust scaler
    scaler = MinMaxScaler()
    prep_df.y = scaler.fit_transform(prep_df.y.values.reshape(-1, 1)).flatten()
    return prep_df


def get_time_series_list(df):
    time_series_list = []
    for col in ["y", "dow", "hour", "month"]:
        time_series_list.append(
            TimeSeries(
                unique_id=col,
                ds=df.ds.values,
                y=df[col].values,
            )
        )
    return time_series_list


HORIZON = 24
LEVELS = [80, 90]
INPUT_SIZE = 2 * HORIZON
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.3

ATTR_METHODS = METHOD_TO_CONSTRUCTOR.keys()

model = NHITS(
    input_size=INPUT_SIZE,
    # inference_input_size=INPUT_SIZE,
    h=HORIZON,
    loss=DistributionLoss(distribution="Normal", level=LEVELS, return_params=True),
    # loss=MQLoss(level=LEVELS),
    max_steps=600,
    random_seed=40,
    # early_stop_patience_steps=5,
    logger=TensorBoardLogger("logs"),
    scaler_type="robust",
    # hist_exog_list=["dow", "hour", "month"],
)
nf_ti_adapter = NhitsNfTiAdapter(model, "h")

loaded_df = load_df()
df = preprocess_df(loaded_df)

# train test split
last_train_idx = int(len(df) * TRAIN_SPLIT)
df_train = df[:last_train_idx]
df_test = df[last_train_idx:]

# save df_test as csv
df_test.to_csv("results_csv/df_test.csv", index=False)

print(df["ds"])
plt.plot(df["ds"][:96], df["y"][:96], color="black")
plt.xlabel("Date", fontsize=18, labelpad=8, color="black")
plt.ylabel("Value", fontsize=18, color="black")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
plt.savefig("results_pdf/sample.pdf", layout="tight")
plt.show()
plt.close()

# train_list = get_time_series_list(df_train)

# nf_ti_adapter.fit_list_exogenous(
#     train_list, "y", val_size=int(VAL_SPLIT * len(df_train.y))
# )
exit()
nf_ti_adapter.fit(df_train.ds, df_train.y, val_size=int(VAL_SPLIT * len(df_train.y)))

start_idx = 126
test_input = df_test.iloc[start_idx : start_idx + INPUT_SIZE]
# test_input_list = get_time_series_list(test_input)

predictions = nf_ti_adapter.predict_plot(
    test_input.ds, test_input.y, test_ds=df_test.ds, test_y=df_test.y
)

for attr_method in ATTR_METHODS:
    target_indices = list(range(len(predictions[f"{model}-loc"])))
    attribution_list, negative_attribution_list, r, rn = nf_ti_adapter.explain(
        attr_method, target_indices, "-loc", test_input.ds.values, test_input.y.values
    )

    plot_attributions(
        attribution_list,
        negative_attribution_list,
        r,
        rn,
        "-loc",
        test_input.ds.values,
        test_input.y.values,
        predictions,
        model,
        attr_method,
    )

    target_indices = list(range(len(predictions[f"{model}-scale"])))
    attribution_list, negative_attribution_list, r, rn = nf_ti_adapter.explain(
        attr_method, target_indices, "-scale", test_input.ds.values, test_input.y.values
    )

    plot_attributions(
        attribution_list,
        negative_attribution_list,
        r,
        rn,
        "-scale",
        test_input.ds.values,
        test_input.y.values,
        predictions,
        model,
        attr_method,
    )
