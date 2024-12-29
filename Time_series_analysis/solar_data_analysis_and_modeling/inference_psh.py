from weather_analysis_utilities import Features, OpenMeteoAPIWrapper
from models import WeatherBasedIrradianceModel, HistoryBasedIrradianceModel
import pickle
import numpy as np


LATITUDE_PSH = 18.852
LONGITUDE_PSH = 98.994


class Inference:
    def __init__(self, daily_init_model_path, hourly_init_model_path):
        self.FEATURES = [Features.TIME.value,
                         Features.SOLAR_POWER.value,
                         Features.SOLAR_IRRADIANCE.value,
                         Features.GHI.value,
                         Features.DHI.value,
                         Features.DNI.value,
                         Features.SHORT_RADIATION.value,
                         Features.DIRECT_RADIATION.value,
                         Features.DIFFUSE_RADIATION.value,
                         Features.DIRECT_NORMAL_RADIATION.value,
                         # Features.TERRESTRIAL_RADIATION.value,
                         Features.TEMPERATURE.value,
                         Features.HUMIDITI.value,
                         Features.RAIN.value,
                         Features.WEATHERCODE.value,
                         Features.CLOUDCOVER.value,
                         Features.CLOUDCOVER_LOW.value,
                         Features.CLOUDCOVER_MID.value,
                         Features.CLOUDCOVER_HIGH.value,
                         Features.WINDSPEED_10.value,
                         # Features.WINDSPEED_80.value,
                         Features.VAPOR_PRESSURE_DEFICIT.value]
        self.latitude = LATITUDE_PSH
        self.longitude = LONGITUDE_PSH
        self.weather_parameters = ["shortwave_radiation",
                                   "direct_radiation",
                                   "diffuse_radiation",
                                   "direct_normal_irradiance"] + self.FEATURES[10:]
        self.weather_data_source = OpenMeteoAPIWrapper(self.latitude, self.longitude, self.weather_parameters)
        with open(daily_init_model_path, "rb") as init_model_file:
            init_model = pickle.load(init_model_file)
            self.weather_based_model = WeatherBasedIrradianceModel(self.weather_parameters, Features.SOLAR_IRRADIANCE.value, init_model)
        with open(hourly_init_model_path, "rb") as init_model_file:
            init_model = pickle.load(init_model_file)
            self.history_based_model = HistoryBasedIrradianceModel(init_model)

    def predict_current_day_mean_irradiance(self) -> float:
        day_weather_data = self.weather_data_source.provide_forecasted_data(1)
        day_weather_data = day_weather_data.resample('24H').mean()
        day_mean_irradiance = self.weather_based_model.irradiance_by_weather(day_weather_data).item()
        return day_mean_irradiance

    def predict_rest_of_day_mean_irradiance(self) -> float:
        # TODO
        return 0

    def predict_next_hour_mean_irradiance(self, history: list[float]) -> float:
        hour_mean_irradiance = self.history_based_model.irradiance_by_history(np.array(history))
        return hour_mean_irradiance


