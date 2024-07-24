import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# Generate the time series data
x = np.arange(0, 500, 1)  # 1000 points
a = 0.5 * np.sin((0.1 * x) + 2)
b = np.cos(0.2 * x)
c = a + b
plt.plot(c)
plt.show()

# Create a DataFrame
data_a = {
    "ds": pd.date_range(start="2020-01-01", periods=len(x), freq="D"),
    "y": a,
    "unique_id": ["ts_a"] * len(x),
}
df_a = pd.DataFrame(data_a)

data_b = {
    "ds": pd.date_range(start="2020-01-01", periods=len(x), freq="D"),
    "y": b,
    "unique_id": ["ts_b"] * len(x),
}
df_b = pd.DataFrame(data_b)

data_c = {
    "ds": pd.date_range(start="2020-01-01", periods=len(x), freq="D"),
    "y": c,
    "unique_id": ["ts_c"] * len(x),
}
df_c = pd.DataFrame(data_c)

df = pd.concat([df_a, df_b, df_c])

# Define and initialize model
model = NHITS(input_size=30, h=48, max_steps=500)
nf = NeuralForecast(models=[model], freq="D")

# Prepare the data for fitting
train_df = df

# Fit the model
nf.fit(train_df)

# Make predictions
forecast_df = nf.predict().reset_index()

# Visualize results for all three input series
plt.figure(figsize=(10, 6))
plt.plot(df_a["ds"], df_a["y"], label="Actual A")
plt.plot(
    forecast_df[forecast_df["unique_id"] == "ts_a"]["ds"],
    forecast_df[forecast_df["unique_id"] == "ts_a"]["NHITS"],
    label="Forecast A",
    linestyle="dashed",
)
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_b["ds"], df_b["y"], label="Actual B")
plt.plot(
    forecast_df[forecast_df["unique_id"] == "ts_b"]["ds"],
    forecast_df[forecast_df["unique_id"] == "ts_b"]["NHITS"],
    label="Forecast B",
    linestyle="dashed",
)
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_c["ds"], df_c["y"], label="Actual C")
plt.plot(
    forecast_df[forecast_df["unique_id"] == "ts_c"]["ds"],
    forecast_df[forecast_df["unique_id"] == "ts_c"]["NHITS"],
    label="Forecast C",
    linestyle="dashed",
)
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
