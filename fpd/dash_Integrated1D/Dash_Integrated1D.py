"""環境"""
# .lab

"""更新履歴"""
# 2025/12/08 作成


"""モジュール読み込み"""
# ファイル操作等
import sys
import os
from datetime import datetime
from pprint import pprint
import logging
import pickle
# import struct
from tqdm import tqdm
import h5py
# import threading
import json
import shutil

# tkinter
from tkinter import filedialog, messagebox, Tk
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1) # tkinterの解像度を上げる

# データ分析ツール
import pandas as pd
import numpy as np
import scipy as sp
import pyFAI
# import math
# from sklearn.linear_model import LinearRegression

# グラフ等作成用
import matplotlib
import matplotlib.pyplot as plt         # 図の作成用
# from PIL import Image as im
# from IPython.display import display, HTML, clear_output, update_display, Image

# 自作モジュール
sys.path.append(r"C:\Users\okaza\pythonenv")
from modules.Mytools.Tools import print_fileinfo, h5_tree, dict_tree, simple_progress_bar, clean_cache_except_logfiles, get_total_size
from modules.Mytools.handle_ipynb import save_pickle, load_pickle, export_html
import modules.Mytools.Settings
import modules.fitXRD as fx
from modules.peakfit import peakfit, pseudoVoigt

# plotly
from dash import Dash, dcc, html, Input, Output # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore

def askopenfilename(filetypes: list) -> str:
    """
    tkinterでファイルダイアログを開く
    """
    window = Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    path = filedialog.askopenfilename(
        parent = window,
        filetypes = filetypes,
    )
    return path

class Widget:
    def __init__(self,
                 path: str):
        
        # hdfファイルのパスを保存
        self.path = path

        # 画像データを生成
        self._make_dataframe()

    def _make_dataframe(self):
        """
        画像データの生成
        
        param:
            resize_width: 動画用低画質画像の横幅
        """
        
        # ファイル情報を出力
        file_stat = print_fileinfo(self.path)
        if file_stat.st_size > 1e8:
            print("File size exceeds 100 MB.")

        # ファイルデータを読み込み
        with h5py.File(self.path, mode = "r") as f:
            self.rad = f["rad"][()] # type: ignore

            stack = []
            for i in range(len(list(f["integrated"].keys()))): # type: ignore
                stack.append(f["integrated/frame = {}".format(i)][()]) # type: ignore

        # Intensityをlogに直す
        self.data = np.stack(stack)

    def make_imshow(self, zmin, zmax):
        
        # figの生成
        fig = px.imshow(
            img = self.data,
            origin = "lower",
            width = 800,
            height = 600,
            aspect = "auto",
            color_continuous_scale = "viridis",
            zmin = zmin,
            zmax = zmax,
            x = self.rad,
            labels = dict(
                x = "Wavelength [nm]",
                y = "Frame",
                animation_frame = "Frame",
                color = "Intensity",
            ),
        )
        return fig


    def create_dash(self):
        
        # Dashアプリの初期化
        app = Dash(__name__)

        # レイアウトの定義
        app.layout = html.Div(children=[
            html.H1(children='Imshow 1D XRD series.'),
            \
            # コントロールエリア
            html.Div(
                [
                    html.H3("設定"),
                    html.Label("強度設定:"),
                    html.Div([
                        html.Label("Min: "),
                        dcc.Input(id='intensity-min',
                                    type='number',
                                    value=np.min(self.data),
                                    step=0.1,
                                    style={'marginRight': '20px'}
                                    ),
                        html.Label("Max: "),
                        dcc.Input(id='intensity-max',
                                    type='number',
                                    value=np.max(self.data),
                                    step=0.1,
                                    style={'marginRight': '20px'}
                                    ),
                    ], style={'marginBottom': '20px'}),
                ],
                style={
                    'padding': '20px',
                    'backgroundColor': '#f9f9f9',
                    'borderRadius': '5px',
                    'maxWidth': '1000px',
                }
            ),
            \
            dcc.Loading(
                dcc.Graph(id = "imshow"),
                type = "cube"
            )
        ])

        @app.callback(
            Output('imshow', 'figure'),
            [Input('intensity-min','value'),
             Input('intensity-max', 'value')])
        def change_zrange(zmin, zmax):
            graph = self.make_imshow(zmin,zmax)
            return graph
        
        return app
    
if __name__ == "__main__":
    path = askopenfilename(filetypes = [("HDF", ".hdf")])
    widget = Widget(path = path)
    app = widget.create_dash()
    app.run(debug = True,
            use_reloader = False,
            port = 8000,
            )