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


class OpenMeteoIrradianceForecasterConfig:

    def __init__(
        self,
        latitude: float,
        longitude: float,
        value_column: str,
        timestamp_column: str,
        open_meteo_requests_cache_expiration_period: int,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.value_column = value_column
        self.timestamp_column = timestamp_column
        self.open_meteo_requests_cache_expiration_period = (
            open_meteo_requests_cache_expiration_period
        )


class OpenMeteoIrradianceForecaster:

    def __init__(self, config: OpenMeteoIrradianceForecasterConfig) -> None:
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
        start_hour = now.floor(self.resolution_string)
        end_hour = start_hour + forecast_size * self.resolution
        irradiance_forecast = self.open_meteo.forecast(start_hour, end_hour)[
            [WeatherParameters.DIRECT_RADIATION.value]
        ]
        irradiance_forecast = irradiance_forecast.rename(
            columns={
                WeatherParameters.DIRECT_RADIATION.value: self.config.value_column
            }
        )
        return irradiance_forecast.squeeze()

    def forecast_hour(
        self,
        now: pd.Timestamp,
    ) -> pd.Series:
        forecast_hour = self.forecast(now, forecast_size=1).values[0]
        return forecast_hour


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

    def format_hour(self, hour_timestamp: pd.Timestamp) -> str:
        year = str(hour_timestamp.year)
        if hour_timestamp.month < 10:
            month = "0" + str(hour_timestamp.month)
        else:
            month = str(hour_timestamp.month)
        if hour_timestamp.day < 10:
            day = "0" + str(hour_timestamp.day)
        else:
            day = str(hour_timestamp.day)
        if hour_timestamp.hour < 10:
            hour = "0" + str(hour_timestamp.hour)
        else:
            hour = str(hour_timestamp.hour)
        if hour_timestamp.minute < 10:
            minute = "0" + str(hour_timestamp.minute)
        else:
            minute = str(hour_timestamp.minute)
        return year + "-" + month + "-" + day + "T" + hour + ":" + minute

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

        hourly_dataframe[self.timestamp_column_name] = hourly_dataframe[
            self.timestamp_column_name
        ] - pd.Timedelta(hours=1)

        hourly_dataframe = hourly_dataframe.set_index(
            self.timestamp_column_name
        )
        hourly_dataframe = hourly_dataframe.tz_convert(self.timezone)

        return hourly_dataframe  # .shift(-1) # .dropna()  # FIXME: lag on one hour, it shows that pick is on 13 o'clok but it must be on 12

    def history(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        params = deepcopy(self.params)
        params["start_date"] = self.format_date(start_date)
        params["end_date"] = self.format_date(end_date)
        responses = self.openmeteo.weather_api(self.url_archive, params=params)
        response = responses[0]
        return self.response_to_data_frame(response)

    """
    def forecast(self, past_days: int, forecast_days: int) -> pd.DataFrame:
        params = deepcopy(self.params)
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days
        responses = self.openmeteo.weather_api(self.url_forecast, params=params)
        response = responses[0]
        return self.response_to_data_frame(response)
    """

    def forecast(
        self, start_hour: pd.Timestamp, end_hour: pd.Timestamp
    ) -> pd.DataFrame:
        params = deepcopy(self.params)
        params["start_hour"] = self.format_hour(start_hour)
        params["end_hour"] = self.format_hour(end_hour)
        responses = self.openmeteo.weather_api(
            self.url_forecast, params=params
        )
        response = responses[0]
        return self.response_to_data_frame(response)
