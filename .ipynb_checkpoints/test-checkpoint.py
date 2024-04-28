from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.utils import AirPassengersDF

nf = NeuralForecast(models=[NBEATS(input_size=24, h=12, max_steps=100)], freq="M")

nf.fit(df=AirPassengersDF)
result = nf.predict()

print(result)
