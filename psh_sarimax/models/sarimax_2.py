import numpy as np
import pandas as pd
import statsmodels.api as sm


class Model():
    def __init__(self, value_column, season_length, forecast_length):
        self._value_column = value_column
        self._season_length = season_length
        self._forecast_length = forecast_length

    def set_parameters(self, p, d, q, P, D, Q):
         self.p = p
         self.d = d
         self.q = q
         self.P = P
         self.D = D
         self.Q = Q

    def fit(self, train_data: pd.DataFrame):
        train_data = train_data.reset_index()[self._value_column]
        self.model_SARIMAX_fited = sm.tsa.statespace.SARIMAX(
            train_data,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self._season_length)
        ).fit(disp=-1)

    def forecast(self, history: np.ndarray) -> np.ndarray:
        new_fitted_model = self.model_SARIMAX_fited.apply(history)
        forecast = new_fitted_model.forecast(self._forecast_length)
        return forecast
