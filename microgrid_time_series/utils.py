from functools import reduce
from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
# pio.renderers.default = "browser" # "svg"


# np.seterr(divide='ignore', invalid='ignore')


class ReporterPlotMaker():
    def __init__(self) -> None:
        self.__figure: Optional[go.Figure] = None
        self.__source_data: Optional[pd.DataFrame] = None

    def read_data(self, data: Optional[pd.DataFrame] = None, data_path: Union[Path, str, None] = None) -> None:
        assert ((data is not None) and (data_path is None))\
             or ((data is None) and (data_path is not None)),\
            "data or data_path must be initialised"
        if data is not None:
            self.__source_data = data.copy()
        else:
            self.__source_data = pd.read_csv(data_path)

    def prepare_figure(self, 
                       primary_data_columns_names, 
                       secondary_data_columns_names, 
                       title="Experiment Telemetry", 
                       height=600, 
                       width=1100,
                       x_axis_column=None,
                       line_style="line") -> None:
        source_data = self.get_data()
        if x_axis_column is not None:
            source_data = source_data.set_index(x_axis_column)
        assert set(primary_data_columns_names + secondary_data_columns_names).issubset(
            source_data.columns), f"not existing data columns: {set(primary_data_columns_names + secondary_data_columns_names) - set(source_data.columns)}"
        self.__figure = make_subplots(specs=[[{"secondary_y": True}]])

        fig_y1 = px.line(source_data[primary_data_columns_names])["data"]
        if line_style == 'scatter':
            fig_y1 = px.scatter(source_data[primary_data_columns_names])["data"]
        for trace in range(len(fig_y1)):
            self.__figure.add_trace(fig_y1[trace], secondary_y=False)

        fig_y2 = px.line(source_data[secondary_data_columns_names])["data"]
        if line_style == 'scatter':
            fig_y2 = px.scatter(source_data[secondary_data_columns_names])["data"]
        for trace in range(len(fig_y2)):
            self.__figure.add_trace(fig_y2[trace], secondary_y=True)

        self.__figure.update_xaxes(title_text="timestamp")
        self.__figure.update_yaxes(title_text="W", secondary_y=False)
        self.__figure.update_yaxes(title_text="percent", secondary_y=True)
        self.__figure.update_layout(showlegend=True, height=height, width=width, title_text=title)

    def get_data(self) -> pd.DataFrame:
        return self.__source_data.copy()

    def plot(self) -> go.Figure:
        return self.__figure

    def save_report(self, filepath: Union[Path, str]) -> None:
        if isinstance(filepath, str):
            if not filepath.endswith(".html"):
                raise ValueError("Invalid file extension")
        elif isinstance(filepath, Path):
            if not filepath.as_posix().endswith(".html"):
                raise ValueError("Invalid file extension")
        else:
            raise ValueError("Invalid filepath type")
        self.__figure.write_html(filepath)
