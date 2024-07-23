from functools import reduce
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



def prepare_figure(dataframe, primary_data_columns_names, secondary_data_columns_names=None, height=600, width=1200, x_axis_column=None, title_text=None):
    dataframe = dataframe.copy()
    if x_axis_column is not None:
        dataframe = dataframe.set_index(x_axis_column)
    if secondary_data_columns_names is not None:
        assert set(primary_data_columns_names + secondary_data_columns_names).issubset(
            dataframe.columns), f"not existing data column. can plot any of {list(dataframe.columns)}"
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    fig_y1 = px.line(dataframe[primary_data_columns_names])["data"]
    for trace in range(len(fig_y1)):
        figure.add_trace(fig_y1[trace], secondary_y=False)
    if secondary_data_columns_names is not None:
        fig_y2 = px.line(dataframe[secondary_data_columns_names])["data"]
        for trace in range(len(fig_y2)):
            figure.add_trace(fig_y2[trace], secondary_y=True)
    figure.update_xaxes()
    figure.update_yaxes(secondary_y=False)
    figure.update_yaxes(secondary_y=True)
    figure.update_layout(showlegend=True, height=height, width=width, title_text=title_text)
    return figure


def plot_telemetry_powers(telemetry, column_name):
    plt.figure(figsize=(20, 10))
    plt.plot(telemetry[column_name], label="column_name")
    plt.legend()
    plt.grid(True)
    plt.show()



