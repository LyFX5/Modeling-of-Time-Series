import pandas as pd
from timezonefinder import TimezoneFinder
from weather_analysis_utilities import Features, OpenMeteoAPIWrapper, Day


class HistoricalDataSource:
    def __init__(self,
                 start_date: Day,
                 end_date: Day,
                 latitude: float,
                 longitude: float,
                 weather_parameters: list[str],
                 timezone_num: int,
                 solar_power_data_path: str,
                 irradiance_data_path: str):
        # self.tz='UTC'
        self.solar_power_data_path = solar_power_data_path
        self.irradiance_data_path = irradiance_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.latitude = latitude
        self.longitude = longitude
        self.weather_parameters = weather_parameters
        self.timezone = TimezoneFinder().certain_timezone_at(lat=self.latitude, lng=self.longitude)
        self.timezone_num = timezone_num
        self.weather_data_source = OpenMeteoAPIWrapper(self.latitude, self.longitude, self.weather_parameters)
        self.__data = self.__load_data()
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
        solar_power_data = solar_power_data.set_index(Features.TIME.value)
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
        irradiance_data = irradiance_data.set_index(Features.TIME.value)
        return irradiance_data

    def __load_weather_data(self):
        return self.weather_data_source.provide_historical_data(self.start_date, self.end_date)

    def __load_data(self):
        solar_power_data = self.__load_solar_power_data()
        irradiance_data = self.__load_irradiance_data()
        weather_data = self.__load_weather_data()
        merged_data = weather_data
        if solar_power_data is not None:
            merged_data = merged_data.merge(solar_power_data, left_index=True, right_index=True)
        if irradiance_data is not None:
            merged_data = merged_data.merge(irradiance_data, left_index=True, right_index=True)
        merged_data = merged_data.dropna()
        float_columns = list(merged_data.columns.values)
        merged_data[float_columns] = merged_data[float_columns].astype(float)
        # merged_data = merged_data.set_index(Features.TIME.value)
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
        for y in range(self.start_date.year, self.end_date.year+1):
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