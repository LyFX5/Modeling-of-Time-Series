import enum
from abc import ABC, abstractmethod
import numpy as np
import pickle
import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from timezonefinder import TimezoneFinder
pd.options.display.max_columns = 100

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from copy import copy, deepcopy

def mare(y: np.ndarray, y_: np.ndarray):
    mean = np.mean(np.abs(y[y != 0] - y_[y != 0]) / y[y != 0])
    # std = np.std(np.abs(y[y != 0] - y_[y != 0]) / y[y != 0])
    min_ = np.min(np.abs(y[y != 0] - y_[y != 0]) / y[y != 0])
    max_ = np.max(np.abs(y[y != 0] - y_[y != 0]) / y[y != 0])
    return [mean, min_, max_]

def mre(y: np.ndarray, y_: np.ndarray) -> float:
    return np.mean((y[y != 0] - y_[y != 0]) / y[y != 0]) * 100

def plot_corr(df):
    for i, method in enumerate(("pearson", "spearman")):
        corr_mat = df.corr(method=method)
        mask = np.zeros_like(corr_mat, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            data=corr_mat,
            mask=mask,
            annot=True,
            fmt='.2f',
            vmax=1,
            center=0,
            square=True,
            linewidth=.5,
            cbar_kws={'shrink': .5},
        )
        plt.title(method)

def show_info(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    df.hist(figsize=(15, 15))
    plot_corr(df)

class Features(Enum):
    TIME = "time"
    SOLAR_POWER = "solar_power"
    SOLAR_IRRADIANCE = "solar_irrad"
    # Irradiance
    GHI = "ghi"
    DHI = "dhi"
    DNI = "dni" # 'direct_normal_irradiance (W/m²)'
    # GHI = DHI + DNIcos(θ)
    SHORT_RADIATION = "shortwave_radiation_instant"
    DIRECT_RADIATION = "direct_radiation_instant"
    DIFFUSE_RADIATION = "diffuse_radiation_instant"
    DIRECT_NORMAL_RADIATION = "direct_normal_irradiance_instant"
    TERRESTRIAL_RADIATION = "terrestrial_radiation_instant"
    # Weather
    TEMPERATURE = "temperature_2m"
    HUMIDITI = "relativehumidity_2m"
    RAIN = "rain"
    WEATHERCODE = "weathercode"
    CLOUDCOVER = "cloudcover"
    CLOUDCOVER_LOW = "cloudcover_low"
    CLOUDCOVER_MID = "cloudcover_mid"
    CLOUDCOVER_HIGH = "cloudcover_high"
    WINDSPEED_10 = "windspeed_10m"
    WINDSPEED_80 = "windspeed_80m"
    VAPOR_PRESSURE_DEFICIT = "vapor_pressure_deficit"

class Units(Enum):
    TIME = enum.auto()
    SOLAR_POWER = "(W)"
    SOLAR_IRRADIANCE = "(W/m²)"
    GHI = "(W/m²)"
    DHI = "(W/m²)"
    DNI = "(W/m²)"  # 'direct_normal_irradiance (W/m²)'
    # GHI = DHI + DNIcos(θ)
    SHORT_RADIATION = "(W/m²)"
    DIRECT_RADIATION = "(W/m²)"
    DIFFUSE_RADIATION = "(W/m²)"
    DIRECT_NORMAL_RADIATION = "(W/m²)"
    TERRESTRIAL_RADIATION = "(W/m²)"
    TEMPERATURE = "(°C)"
    HUMIDITI = "(%)"
    RAIN = "(mm)"
    WEATHERCODE = "(wmo code)"
    CLOUDCOVER = "(%)"
    CLOUDCOVER_LOW = "(%)"
    CLOUDCOVER_MID = "(%)"
    CLOUDCOVER_HIGH = "(%)"
    WINDSPEED_10 = "(km/h)"
    WINDSPEED_80 = "(km/h)"
    VAPOR_PRESSURE_DEFICIT = "(kPa)"

FEATURES = [Features.TIME.value,
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

class Day:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

class HistoricalDataSource:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 latitude: float,
                 longitude: float,
                 timezone_num: int,
                 solar_power_data_path: str,
                 irradiance_data_path: str,
                 postfix_data_flag: bool = False):
        # self.tz='UTC'
        self.solar_power_data_path = solar_power_data_path
        self.irradiance_data_path = irradiance_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = TimezoneFinder().certain_timezone_at(lat=self.latitude, lng=self.longitude)
        self.timezone_num = timezone_num
        self.weather_api_url = f"https://archive-api.open-meteo.com/v1/archive?" \
                       f"latitude={self.latitude}&longitude={self.longitude}&start_date={self.start_date}&end_date={self.end_date}&" \
                       f"hourly="
        self.weather_parameters = ["shortwave_radiation",
                           "direct_radiation",
                           "diffuse_radiation",
                           "direct_normal_irradiance"] + FEATURES[10:]
        self.__data = self.__load_data()
        self.__postfix_data = None
        if postfix_data_flag:
            self.__postfix_data = self.__prepare_day_postfix_data()

    def __load_solar_power_data(self):
        if self.solar_power_data_path is None:
            return None
        data = pd.read_csv(self.solar_power_data_path, parse_dates=[Features.TIME.value])
        solar_power_data = data[[Features.TIME.value, Features.SOLAR_POWER.value]]
        solar_power_data[Features.SOLAR_POWER.value].clip(lower=0, inplace=True)
        solar_power_data[Features.TIME.value] = pd.to_datetime(solar_power_data[Features.TIME.value], utc=True)
        # solar_power_data = solar_power_data.assign(time=solar_power_data.time.dt.tz_convert(self.timezone_num))
        solar_power_data.solar_power = solar_power_data.solar_power.shift(self.timezone_num // 3600)
        return solar_power_data

    def __load_irradiance_data(self):
        if self.irradiance_data_path is None:
            return None
        irradiance_data = pd.read_csv(self.irradiance_data_path, parse_dates=[Features.TIME.value])
        irradiance_data = irradiance_data[[Features.TIME.value, Features.SOLAR_IRRADIANCE.value]]
        irradiance_data[Features.SOLAR_IRRADIANCE.value].clip(lower=0, inplace=True)
        irradiance_data[Features.TIME.value] = pd.to_datetime(irradiance_data[Features.TIME.value], utc=True)
        # irradiance_data = irradiance_data.assign(time=irradiance_data.time.dt.tz_convert(self.timezone_num))
        irradiance_data.solar_irrad = irradiance_data.solar_irrad.shift(self.timezone_num // 3600)
        return irradiance_data

    def __load_weather_data(self):
        api_url = self.weather_api_url
        for i in range(len(self.weather_parameters)):
            api_url += self.weather_parameters[i]
            if i < len(self.weather_parameters) - 1:
                api_url += ","
        api_url += f"&timezone={self.timezone}&min={self.start_date}&max={self.end_date}"
        weather_data = pd.read_json(api_url)
        weather_data_df = pd.DataFrame()
        weather_data_df[Features.TIME.value] = weather_data["hourly"][Features.TIME.value]
        for parameter in self.weather_parameters:
            weather_data_df[parameter] = weather_data["hourly"][parameter]
        weather_data_df[Features.TIME.value] = pd.to_datetime(weather_data_df[Features.TIME.value], utc=True)
        # weather_data_df = weather_data_df.assign(time=weather_data_df.time.dt.tz_convert(self.timezone_num))
        return weather_data_df

    def __load_data(self):
        solar_power_data = self.__load_solar_power_data()
        irradiance_data = self.__load_irradiance_data()
        weather_data = self.__load_weather_data()
        merged_data = weather_data
        if solar_power_data is not None:
            merged_data = merged_data.merge(solar_power_data, on=Features.TIME.value)
        if irradiance_data is not None:
            merged_data = merged_data.merge(irradiance_data, on=Features.TIME.value)
        merged_data = merged_data.dropna()
        float_columns = list(merged_data.columns.values)
        float_columns.remove(Features.TIME.value)
        merged_data[float_columns] = merged_data[float_columns].astype(float)
        merged_data = merged_data.set_index(Features.TIME.value)
        return merged_data

    def read_day(self, year, month, day) -> pd.DataFrame:
        day = self.__data[((self.__data.index.year == year) &
                         (self.__data.index.month == month) &
                         (self.__data.index.day == day))]
        # day = day[day[Features.SOLAR_POWER.value] != 0.]
        return day

    def read_past_part_of_day(self, year, month, day, hour) -> pd.DataFrame:
        # TODO can use df.head(n)
        day = self.read_day(year, month, day)
        assert hour in day.index.hour[1:], f"The {hour=} is out of day range {day.index.hour=}."
        past_part_of_day = day[(day.index.hour < hour)]
        return past_part_of_day

    def read_rest_part_of_day(self, year, month, day, hour) -> pd.DataFrame:
        # TODO can use df.tail(n)
        day = self.read_day(year, month, day)
        assert hour in day.index.hour, f"The {hour=} is out of day range {day.index.hour=}."
        rest_part_of_day = day[(day.index.hour >= hour)]
        return rest_part_of_day

    def __prepare_day_postfix_data(self):
        day_postfix_data = pd.DataFrame()
        start_timestamp = pd.Timestamp(self.start_date)
        end_timestamp = pd.Timestamp(self.end_date)
        for y in range(start_timestamp.year, end_timestamp.year+1):
            for m in range(1, 13):
                for d in range(1, 31):
                    one_day = self.read_day(y, m, d)
                    if not one_day.empty:
                        one_day = one_day.reset_index()
                        one_day = one_day.drop(columns=[Features.TIME.value])
                        for h in range(24, 0, -1):
                            current_h = 24 - h
                            tail_df = pd.DataFrame(columns=(list(self.__data.columns)))
                            tail_mean = one_day.tail(h).mean()
                            tail_df = pd.concat([tail_df,
                                                pd.DataFrame.from_dict({0: list(tail_mean.values)}, orient='index',
                                                                       columns=list(tail_mean.index))])
                            tail_df[Features.TIME.value] = pd.Timestamp(year=y, month=m, day=d, hour=current_h, tz='UTC')
                            tail_df = tail_df.set_index(Features.TIME.value)
                            # print(tail_df)
                            if day_postfix_data.empty:
                                day_postfix_data = tail_df
                            else:
                                day_postfix_data = pd.concat([day_postfix_data, tail_df])
        return day_postfix_data

    @property
    def postfix_data(self) -> pd.DataFrame:
        return self.__postfix_data

    @property
    def data(self) -> pd.DataFrame:
        return self.__data


class WeatherForecaster:
    def __init__(self,
                 latitude,
                 longitude):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = TimezoneFinder().certain_timezone_at(lat=self.latitude, lng=self.longitude)
        self.weather_api_url = f"https://api.open-meteo.com/v1/forecast?latitude={self.latitude}&longitude={self.longitude}" \
                       f"&timezone={self.timezone}&hourly="

    def hourly_forecast_for_n_days_ahead(self,
                                         weather_parameters: list[str],
                                         n_days: int) -> pd.DataFrame:
        assert 0 < n_days < 7
        api_url = self.weather_api_url
        for i in range(len(weather_parameters)):
            api_url += weather_parameters[i]
            if i < len(weather_parameters)-1:
                api_url += ","
        api_url += f"&forecast_days={n_days}"
        weather_forecast = pd.read_json(api_url)
        weather_forecast_df = pd.DataFrame()
        weather_forecast_df[Features.TIME.value] = weather_forecast["hourly"][Features.TIME.value]
        for parameter in weather_parameters:
            weather_forecast_df[parameter] = weather_forecast["hourly"][parameter]
        weather_forecast_df[Features.TIME.value] = pd.to_datetime(weather_forecast_df[Features.TIME.value], utc=True)
        # weather_forecast_df = weather_forecast_df.assign(time=weather_forecast_df.time.dt.tz_convert(self.timezone))
        weather_forecast_df = weather_forecast_df.set_index(Features.TIME.value)
        # print(weather_forecast_df.head())
        return weather_forecast_df


class WeatherBasedDailyMeanForecaster:
    def __init__(self,
                 historical_data_source: HistoricalDataSource,
                 weather_forecaster: WeatherForecaster,
                 features: list = None,
                 target: str = None,
                 pretrained_model=None,
                 whole_day_mean_forecast: bool = True):
        self.historical_data_source = historical_data_source
        self.weather_forecaster = weather_forecaster
        self.features = features
        if self.features is None:
            self.features = self.historical_data_source.weather_parameters
        self.target = target
        if self.target is None:
            self.target = Features.SOLAR_POWER.value
        self.__model = pretrained_model
        if self.__model is None:
            self.evaluation_results = self.__prepare_init_model(whole_day_mean_forecast)
        self.__actual_model_index = 0
        self.__models_dict = {self.__actual_model_index: self.__model}

    def add_forecast_model(self, model_index, new_model):
        self.__models_dict[model_index] = new_model

    def change_actual_model(self, model_index: int):
        assert model_index in self.__models_dict.keys(), f"available indexes are {self.__models_dict.keys()}"
        self.__actual_model_index = model_index

    @property
    def model(self):
        return self.__models_dict[self.__actual_model_index]

    def __normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        # TODO нормализация, добавление средней мощности за предыдущую часть дня
        return data

    def __split_data(self, data: pd.DataFrame, features_columns, target_column, test_size=0.35,):
        data = data.dropna()
        x_train, x_test, y_train, y_test = \
            train_test_split(data[features_columns], data[target_column], test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test

    def __resample_data(self, data: pd.DataFrame):
        return data.resample('24H').mean()

    def __prepare_init_model(self, whole_day_mean_forecast):
        self.__model = LGBMRegressor()
        if whole_day_mean_forecast:
            data = self.historical_data_source.data
            # data = self.__normalize(data)
            data = self.__resample_data(data)
        else:
            data = self.historical_data_source.postfix_data
        data = data[[self.target] + self.features]
        x_train, x_test, y_train, y_test = self.__split_data(data,
                                                             features_columns = self.features,
                                                             target_column = self.target)
        self.__model.fit(x_train, y_train)
        prediction = self.__model.predict(x_test)
        MSE = mean_squared_error(y_test.values, prediction)
        R2 = r2_score(y_test.values, prediction)
        RMSE = MSE ** 0.5
        MAE = np.mean(np.abs(y_test.values - prediction))
        MRE = mre(y_test.values, prediction)
        [MARE_mean, MARE_min, MARE_max] = mare(y_test.values, prediction)
        return {"data":
                    {"resampled_data": data,
                     "x_train": x_train,
                     "x_test": x_test,
                     "y_train": y_train,
                     "y_test": y_test},
                "prediction": prediction,
                "metrics":
                    {"MAE": MAE,
                     "MSE": MSE,
                     "RMSE": RMSE,
                     "R2": R2,
                     "MRE": MRE,
                     "MARE_mean": MARE_mean,
                     "MARE_min": MARE_min,
                     "MARE_max": MARE_max}
                }

    def forecast_day_mean(self, day_weather_forecast: pd.DataFrame) -> float :
        day_weather_forecast_norm = self.__normalize(day_weather_forecast)
        day_weather_forecast_norm_mean = day_weather_forecast_norm.resample('24H').mean()
        return self.model.predict(day_weather_forecast_norm_mean).item()

    def forecast_current_day_mean(self) -> float:
        day_weather_forecast = self.weather_forecaster.hourly_forecast_for_n_days_ahead(weather_parameters=self.historical_data_source.weather_parameters,
                                                                                        n_days=1)
        return self.forecast_day_mean(day_weather_forecast)

    def validate_on_sample(self, days: list[Day]) -> [pd.DataFrame, list[float]]:
        days_data = None
        error = []
        for i in range(len(days)):
            day_data = self.historical_data_source.read_day(days[i].year,
                                                            days[i].month,
                                                            days[i].day)
            features = self.historical_data_source.weather_parameters
            day_mean_forecast = self.forecast_day_mean(day_data[features])
            assert day_mean_forecast is not None
            day_mean = day_data[self.target].mean()
            day_data[f"{self.target}_day_mean"] = day_mean * np.ones(len(day_data.index))
            day_data[f"{self.target}_day_mean_forecast"] = day_mean_forecast * np.ones(len(day_data.index))
            if i == 0:
                days_data = day_data
            else:
                days_data = pd.concat([days_data, day_data])
            error.append(day_mean-day_mean_forecast)
        return [days_data, error]

    def save_model(self, file_name):
        with open(file_name, "wb") as model_file:
            pickle.dump(self.__model, model_file)


class HistoryBasedHourlyForecaster:
    def __init__(self,
                 historical_data_source: HistoricalDataSource,
                 target: str,
                 target_data_path: str = None,
                 pretrained_model = None,
                 forecast_length: int = 1):
        self.historical_data_source = historical_data_source
        self.target = target
        self.target_data_path = target_data_path
        if self.target_data_path is None:
            self.target_data_path = f"{self.target}_data.csv"
            self.__prepare_historical_target_data()
        self.forecast_length = forecast_length
        self.__model = pretrained_model
        if self.__model is None:
            self.evaluation_results = self.__prepare_init_model()
        self.__actual_model_index = 0
        self.__models_dict = {self.__actual_model_index: self.__model}
        self.history_length = 12 # hours (Ridge is expecting 10 features as input.)

    def add_model(self, model_index, new_model):
        self.__models_dict[model_index] = new_model

    def change_actual_model(self, model_index: int):
        assert model_index in self.__models_dict.keys(), f"available indexes are {self.__models_dict.keys()}"
        self.__actual_model_index = model_index

    @property
    def model(self):
        return self.__models_dict[self.__actual_model_index]

    def __prepare_historical_target_data(self) -> pd.DataFrame:
        assert self.target_data_path is not None
        return self.historical_data_source.data[self.target].to_csv(self.target_data_path)

    def __prepare_init_model(self):
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.forecast_length))
        train_input = InputData.from_csv_time_series(task=task,
                                                     file_path=self.target_data_path,
                                                     delimiter=',',
                                                     target_column=self.target)
        train_data, test_data = train_test_data_setup(train_input)
        self.__model = Fedot(problem='ts_forecasting', task_params=task.task_params)
        pipeline = PipelineBuilder() \
            .add_sequence("glm", branch_idx=0) \
            .add_sequence("lagged", "ridge", branch_idx=1).join_branches("ridge").build()
        pipeline = self.__model.fit(train_data, predefined_model=pipeline)
        out_of_sample_forecast = self.__model.forecast(test_data)
        metrics = self.__model.get_metrics(target=test_data.target, metric_names=['rmse', 'mae', 'mape', 'r2'])
        return {
                "train_data": train_data,
                "test_data": test_data,
                "pipeline": pipeline,
                "out_of_sample_forecast": out_of_sample_forecast,
                "metrics": metrics,
                }

    def forecast(self, history: np.ndarray) -> float:
        return self.__model.forecast(pre_history=history).item()

    def validate_on_sample(self, sample_data: pd.DataFrame):
        lagged_data = pd.DataFrame()
        for i in range(self.history_length, 0, -1):
            lagged_data[f"{self.target}_d{i}"] = sample_data.shift(i)
        lagged_data[self.target] = sample_data
        predicted = []
        i = 0
        for r in lagged_data.values:
            if i < self.history_length:
                predict = r[-1]
            else:
                predict = self.forecast(r[:self.history_length])
            predicted.append(predict)
            i += 1
        predicted = np.array(predicted)
        lagged_data[f"{self.target}_forecast"] = predicted
        lagged_data = lagged_data.dropna()
        return lagged_data

    def save_model(self, file_name):
        with open(file_name, "wb") as model_file:
            pickle.dump(self.__model, model_file)




