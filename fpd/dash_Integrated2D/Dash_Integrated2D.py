"""環境"""
# .lab

"""更新履歴"""
# 2025/12/10 配布用に編集
# 2025/12/10 低画質モードと高画質モードの作成
# 2025/12/09 低画質モードと高画質モードの作成（未完成）
# 2025/12/08 作成

"""モジュール読み込み"""
# ファイル操作等
import sys
import os
from datetime import datetime
import h5py

# tkinter
from tkinter import filedialog, messagebox, Tk
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1) # tkinterの解像度を上げる

# データ分析ツール
import numpy as np

# 画像リサイズ用
import cv2 # type: ignore

# plotly
from dash import Dash, dcc, html, Input, Output, State # type: ignore
import plotly # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore

# ファイル情報を出力する関数
def print_fileinfo(filename):
    file_stat = os.stat(filename)
    print("")
    print('File name: {}'.format(os.path.abspath(filename)))
    print('File size: {:,} bites'.format(file_stat.st_size))
    print('Last update time: {}'.format(datetime.fromtimestamp(file_stat.st_mtime)))
    print("")
    return file_stat

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
                 path: str,
                 resize_width: int = 200):
        
        # hdfファイルのパスを保存
        self.path = path

        # 画像データを生成
        self._make_dataframe(resize_width)

        # ラベルの管理
        self.label_lowq = "Animation (Low res)"
        self.label_highq = "Frame-respective (high res)"

    def _make_dataframe(self, resize_width:int):
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
            self.azim = f["azim"][()] # type: ignore

            stack = []
            for i in range(len(list(f["integrated"].keys()))): # type: ignore
                stack.append(f["integrated/frame = {}".format(i)][()]) # type: ignore

        # Intensityをlogに直す
        stack_log = np.stack(stack)
        stack_log[stack_log < 1] = 1
        stack_log = np.log(stack_log)
        self.data = stack_log
        print("Raw: {:.2f} MB".format(sys.getsizeof(self.data) * 1e-6))
        print(self.data[0].shape)

        # 低画質の画像を生成する
        width = resize_width
        h,w = stack[0].shape[:2]
        height = round(h*(width)/w)
        stack_resized = list(map(lambda img: cv2.resize(img, dsize = (width, height)),stack))
        self.rad_resized = np.linspace(np.min(self.rad), np.max(self.rad), width) # type: ignore
        self.azim_resized = np.linspace(np.min(self.azim), np.max(self.azim), height) # type: ignore

        # Intensityをlogに直す
        resized_log = np.stack(stack_resized)
        resized_log[resized_log < 1] = 1
        resized_log = np.log(resized_log)
        self.resized = resized_log
        print("Resized: {:.2f} MB".format(sys.getsizeof(resized_log) * 1e-6))
    
    def _graph_init(self, ids: dict):

        graphs = []

        # 動画用
        graphs.append(
            html.Div(
                [
                    dcc.Graph(id = ids["graph-lowq"],
                              figure = self._make_graph_lowq(
                                  zmin = self.data.min(),
                                  zmax = self.data.max(),
                                  frame = 0,
                                  )
                              )
                ],
                id = ids["window-lowq"],
                style = {'display': "none"}
            )
        )

        num_frames = self.data.shape[0]
        # 静止画用
        graphs.append(
            html.Div(
                [
                    html.Label("フレーム番号:"),
                    html.Div(
                        dcc.Dropdown(
                            id=ids["dropdown"],
                            value=0,
                            clearable=False,
                            options = [
                                {'label': str(i),
                                'value': i
                                } for i in range(num_frames)
                            ], 
                        ),
                        style = {
                            'width': '10%',
                            'margin-left': 10,
                            'display': 'inline-block',
                            'verticalAlign': 'middle',
                        },
                    ),
                    dcc.Loading(
                        dcc.Graph(id = ids["graph-highq"],
                                  figure = self._make_graph_highq(
                                      zmin = self.resized.min(),
                                      zmax = self.resized.max(),
                                      frame = 0,
                                      )
                                  ),
                        type = "cube"
                    )
                ],
                id = ids["window-highq"],
                style={'display': "none"}
            )
        )

        return graphs

    def _make_graph_lowq(self, zmin, zmax, frame):

        num_frames = self.resized.shape[0]

        # 1. ベースとなるFigureを作成し、最初のフレーム(t=0)を追加
        fig = go.Figure(
            data=[go.Heatmap(
                z=self.resized[0],
                zmin=zmin,
                zmax=zmax,
                x = self.rad_resized,
                y = self.azim_resized,
                colorscale='Viridis',
                colorbar=dict(title='Intensity') # カラーバーのラベル
            )]
        )

        # 2. フレーム(Frames)のリストを作成
        # 各時点のデータをgo.Frameオブジェクトとして定義します
        frames_list = [
            go.Frame(
                data=[go.Heatmap(z=self.resized[k])],
                name=str(k) # フレーム名（スライダーとの紐付けに重要）
            )
            for k in range(num_frames)
        ]
        fig.frames = frames_list

        # 3. アニメーション設定 (再生ボタン)
        updatemenus = [dict(
            type='buttons',
            direction = 'left', # 横並びの場合 'left', 縦並びの場合 'down'
            showactive=False,
            y=0,
            x=0,
            xanchor='left',
            yanchor='top',
            pad=dict(t=65, r=0),
            buttons=[
                dict(
                    label='▶︎',
                    method='animate',
                    args=[None,
                          dict(frame=dict(duration=100,
                                          redraw=True
                                          ),
                               fromcurrent=True,
                               mode='immediate'
                               )
                          ]
                ),
                dict(
                    label='⏸',
                    method='animate',
                    args=[[None],
                          dict(frame=dict(duration=0,
                                          redraw=False
                                          ),
                               mode='immediate',
                               transition=dict(duration=0)
                               )
                          ]
                )
            ]
        )]

        # 4. スライダー設定
        sliders = [dict(
            steps=[dict(
                method='animate',
                args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=str(k),
            ) for k in range(num_frames)],
            ticklen = 0,
            minorticklen = 0,
            active=0,
            y=0,
            x=0, # スライダーの開始位置
            len=1, # スライダーの長さ
            xanchor='left',
            yanchor='top',
            pad=dict(b=10, t=50, l=100),
            currentvalue=dict(prefix='Frame = ', visible=True, xanchor='left'),
        )]

        # 5. レイアウトの更新
        fig.update_layout(
            title="動画表示（低画質モード）",
            updatemenus=updatemenus,
            sliders=sliders,
            height=600,
            width=800,
            xaxis = {'title': "Diffraction angle (2theta) [deg.]"},
            yaxis = {'title': "Azimuthal angle (phi) [deg.]"}
        )

        # 6. フレームの更新
        with fig.batch_update():
            for i, trace in enumerate(fig.frames[frame].data):
                # 既存のfig.data[i]に対して、フレームのデータで上書き更新をかける
                fig.data[i].update(trace)
            
            # スライダーの位置更新
            fig.layout.sliders[0].active = frame

        return fig

    def _make_graph_highq(self, zmin, zmax, frame):

        img = self.data[frame]
                    
        fig = go.Figure(
            data=[go.Heatmap(
                z=img,
                zmin=zmin,
                zmax=zmax,
                colorscale='Viridis',
                colorbar=dict(title='Intensity'),
                x = self.rad,
                y = self.azim,
            )]
        )
        fig.update_layout(
            title = "静止画表示（高画質モード）",
            height=600,
            width=800,
            xaxis = {'title': "Diffraction angle (2theta) [deg.]"},
            yaxis = {'title': "Azimuthal angle (phi) [deg.]"},
        )
        return fig

    def create_dash(self):

        # --- 2. Dashアプリケーションのセットアップ ---
        app = Dash(__name__)

        app.layout = html.Div([
            html.H1(
                "動画解析用GUI (developed by Gemini)",
            ),
            \
            html.Div([
                # コントロールエリア
                html.Div(
                    [
                        html.H3("設定"),
                        html.Label("表示するグラフの選択:"),
                        dcc.RadioItems(
                            id='selection',
                            options=[self.label_lowq, self.label_highq],
                            value=self.label_lowq,
                        ),
                        html.Hr(),
                        html.Label("強度設定:"),
                        html.Div([
                            html.Label("Min: "),
                            dcc.Input(id='intensity-min',
                                      type='number',
                                      value=np.min(self.resized),
                                      step=0.1,
                                      style={'marginRight': '20px'}
                                      ),
                            html.Label("Max: "),
                            dcc.Input(id='intensity-max',
                                      type='number',
                                      value=np.max(self.resized),
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
                html.Hr(),
                \
                # プロットエリア
                dcc.Loading(
                    self._graph_init(
                        ids = {'graph-highq': "graph-highq",
                               'window-highq': "window-highq",
                               'window-lowq': "window-lowq",
                               'graph-lowq': "graph-lowq",
                               'dropdown': "frame-dropdown"}
                    ),
                    type = "cube"
                ),
                \
                dcc.Store(id = "intermediate-values")
            ])
        ])

        # --- 3. コールバック定義 ---

        @app.callback(
            [Output('window-highq', 'style'),
             Output('window-lowq', 'style')],
             Input('selection', 'value'))
        def toggle_container_visibility(selected_window):
            visible_style = {
                'display': "block",
                'border': "1px solid #ddd"
            }
            hidden_style = {
                'display': "none"
            }
            if selected_window == self.label_highq:
                return visible_style, hidden_style
            else:
                return hidden_style, visible_style
            
        @app.callback(
            [Output('graph-highq', 'figure'),
             Output('graph-lowq', 'figure')],
            [Input('intensity-min','value'),
             Input('intensity-max', 'value'),
             Input('selection', 'value')],
             Input('frame-dropdown', 'value'))
        def change_zrange(zmin, zmax, selected_window, frame):
            if selected_window == self.label_highq:

                graph = self._make_graph_highq(zmin, zmax, frame)
                return graph, None
            else:
                graph = self._make_graph_lowq(zmin, zmax, frame)
                return None, graph

        return app
    
if __name__ == "__main__":
    path = askopenfilename(filetypes = [("HDF", ".hdf")])
    widget = Widget(path = path, resize_width = 150)
    app = widget.create_dash()
    app.run(debug = True,
            use_reloader = False,
            port = 8025)