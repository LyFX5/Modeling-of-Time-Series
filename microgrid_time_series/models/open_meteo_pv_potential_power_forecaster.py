from copy import deepcopy
from enum import Enum
from typing import List

import openmeteo_requests  # type: ignore
import pandas as pd
import requests_cache
from retry_requests import retry  # type: ignore
from timezonefinder import TimezoneFinder


class WeatherParameters(Enum):

    SHORT_RADIATION = "shortwave_radiation"
    DIRECT_RADIATION = "direct_radiation"
    DIFFUSE_RADIATION = "diffuse_radiation"
    DIRECT_NORMAL_RADIATION = "direct_normal_irradiance"
    TERRESTRIAL_RADIATION = "terrestrial_radiation"


class OpenMeteoPVPotentialPowerForecasterConfig:

    def __init__(
        self,
        latitude: float,
        longitude: float,
        value_column: str,
        timestamp_column: str,
        irradiance_to_pv_scaler: float,
        open_meteo_requests_cache_expiration_period: int,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.value_column = value_column
        self.timestamp_column = timestamp_column
        self.irradiance_to_pv_scaler = irradiance_to_pv_scaler
        self.open_meteo_requests_cache_expiration_period = (
            open_meteo_requests_cache_expiration_period
        )


class OpenMeteoPVPotentialPowerForecaster:

    def __init__(self, config: OpenMeteoPVPotentialPowerForecasterConfig) -> None:
        self.config = config
        self.open_meteo = OpenMeteoAPIHandler(
            self.config.latitude,
            self.config.longitude,
            self.config.timestamp_column,
            [
                WeatherParameters.SHORT_RADIATION.value,
                WeatherParameters.DIRECT_RADIATION.value,
                WeatherParameters.DIFFUSE_RADIATION.value,
                WeatherParameters.DIRECT_NORMAL_RADIATION.value,
                WeatherParameters.TERRESTRIAL_RADIATION.value,
            ],
            self.config.open_meteo_requests_cache_expiration_period,
        )

    @property
    def resolution(self) -> pd.Timedelta:
        return pd.Timedelta(1, "h")

    @property
    def resolution_string(self) -> str:
        return self.resolution.resolution_string

    def forecast(
        self,
        now: pd.Timestamp,
        forecast_size: int,
    ) -> pd.Series:
        # assert (0 <= current_hour < 23) and (0 < forecast_size < 24), f"not (0 <= {current_hour=} < 24) and (0 < {forecast_size=} < 24)" # returns until 22 hour of current day
        irradiance_forecast = self.open_meteo.forecast(past_days=0, forecast_days=2)[
            [WeatherParameters.DIRECT_RADIATION.value]
        ]
        forecast = self.config.irradiance_to_pv_scaler * irradiance_forecast.rename(
            columns={WeatherParameters.DIRECT_RADIATION.value: self.config.value_column}
        )
        current_hour_timestamp = now.floor(self.resolution_string)
        forecast = forecast[
            current_hour_timestamp : current_hour_timestamp  # type: ignore[misc]
            + (forecast_size - 1) * self.resolution
        ]
        return forecast.squeeze()


class OpenMeteoAPIHandler:

    def __init__(
        self,
        latitude,
        longitude,
        timestamp_column_name,
        weather_parameters: List[str],
        requests_cache_expiration_period,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timestamp_column_name = timestamp_column_name
        self.weather_parameters = [param for param in weather_parameters]
        self.timezone = TimezoneFinder().certain_timezone_at(
            lat=self.latitude, lng=self.longitude
        )
        cache_session = requests_cache.CachedSession(
            ".cache", expire_after=requests_cache_expiration_period
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url_archive = "https://archive-api.open-meteo.com/v1/archive"
        self.url_forecast = "https://api.open-meteo.com/v1/forecast"
        self.params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "hourly": self.weather_parameters,
        }

    def format_date(self, date: pd.Timestamp) -> str:
        year = str(date.year)
        if date.month < 10:
            month = "0" + str(date.month)
        else:
            month = str(date.month)
        if date.day < 10:
            day = "0" + str(date.day)
        else:
            day = str(date.day)
        return year + "-" + month + "-" + day

    def response_to_data_frame(self, response) -> pd.DataFrame:
        hourly = response.Hourly()
        hourly_data = {
            self.timestamp_column_name: pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }  # time range in UTC
        for i in range(len(self.weather_parameters)):
            hourly_data[self.weather_parameters[i]] = hourly.Variables(
                i
            ).ValuesAsNumpy()
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframe = hourly_dataframe.set_index(self.timestamp_column_name)
        return hourly_dataframe  # FIXME: lag on one hour, it shows that pick is on 13 o'clok but it must be on 12

    def history(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        params = deepcopy(self.params)
        params["start_date"] = self.format_date(start_date)
        params["end_date"] = self.format_date(end_date)
        responses = self.openmeteo.weather_api(self.url_archive, params=params)
        response = responses[0]
        return self.response_to_data_frame(response)

    def forecast(self, past_days: int, forecast_days: int) -> pd.DataFrame:
        params = deepcopy(self.params)
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days
        responses = self.openmeteo.weather_api(self.url_forecast, params=params)
        response = responses[0]
        return self.response_to_data_frame(response)
