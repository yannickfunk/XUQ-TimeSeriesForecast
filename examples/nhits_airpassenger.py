import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.utils import AirPassengersDF

# Split data and declare panel dataset
Y_df = AirPassengersDF

split_day = "1955-08-31"
Y_train_df = Y_df[Y_df.ds <= split_day]
Y_test_df = Y_df[Y_df.ds > split_day]
horizon = len(Y_test_df)

nf = NeuralForecast(
    models=[NHITS(input_size=64, h=horizon, start_padding_enabled=True)],
    freq="ME",
)
nf.fit(df=Y_train_df)
predictions = nf.predict(df=Y_train_df)


plt.plot(Y_df.ds, Y_df.y, label="Ground Truth")
plt.plot(predictions.ds, predictions.NHITS, label="Predictions")
plt.legend()
plt.show()
