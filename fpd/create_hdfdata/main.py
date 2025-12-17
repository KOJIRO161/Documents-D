"""環境"""
# .lab

"""更新履歴"""
# 2025-12-17 integrated1dを追加しました
# 2025-12-17 生データを作成しなくても積算できるようにしました。
# 2025-12-01 作成

"""モジュール読み込み"""
# ファイル操作等
import sys
import os
# from datetime import datetime
# from pprint import pprint
# import logging
# import pickle
# import struct
from tqdm import tqdm
import h5py
# import threading
# import json
# import shutil

# tkinter
from tkinter import filedialog, messagebox, Tk
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1) # tkinterの解像度を上げる

# データ分析ツール
import pandas as pd
import numpy as np
import scipy as sp
import pyFAI

# グラフ等作成用
# import matplotlib
# import matplotlib.pyplot as plt         # 図の作成用
# from PIL import Image as im
# import cv2
# from IPython.display import display, HTML, clear_output, update_display, Image

# 自作モジュール
sys.path.append(r"C:\Users\okaza\pythonenv")
from modules.Mytools.Tools import print_fileinfo, h5_tree, dict_tree, simple_progress_bar, clean_cache_except_logfiles, get_total_size
# from modules.Mytools.handle_ipynb import save_pickle, load_pickle, export_html
# import modules.Mytools.Settings
# import modules.fitXRD as fx
# from modules.peakfit import peakfit, pseudoVoigt

# fpdファイル取扱い用
import threading
import concurrent.futures as confu
from modules.Mytools.Tools import his2array

class Create_Hdfdata:
    def __init__(self,
                 name = "",
                 clear_cachedir: bool = False,
                 cachedir: str= "",
                 log: bool = False):
        
        self.name = name
        self.cachedir: str = os.path.join(os.getcwd(), ".cache") if (cachedir == "") else cachedir
        self.log: bool = log
        self.version: str = "1.0"

        # 内部変数
        self._set_init_values()

        # cachedirの作成
        self.cachedir: str = os.path.join(os.getcwd(), ".cache")
        if clear_cachedir:
            clean_cache_except_logfiles(self.cachedir)
        if self.log:
            print("name: " + self.name)
            print("cachedir: " + self.cachedir)
            print("log: {}".format(self.log))
            print("version: {}".format(self.version))

    def _set_init_values(self):

        self.filelist: list = []

    def askopenfilename(self,
                        filetypes: list) -> str:
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
    
    def get_rawdata(self) -> str:

        # 生データをhdfで保存
        rawhdf = os.path.join(self.cachedir, self.name + "_raw.hdf")

        with h5py.File(rawhdf, mode = "w"):
            pass

        for i, filename in enumerate(self.filelist):
            d = his2array(filename)
            with h5py.File(rawhdf, mode = "r+") as f:
                f.create_dataset(
                    name = "frame = {}".format(i),
                    data = d,
                    shape = d.shape,
                    dtype = d.dtype,
                )
            if self.log:
                simple_progress_bar(i+1, len(self.filelist))

        if self.log:
            print("\nRaw data saved.")
            with h5py.File(rawhdf, mode = "r") as f:
                h5_tree(f)

        # 出力
        return rawhdf
        
    def integrate2D(self,
                    poni: str,
                    npt_rad: int,
                    npt_azim: int,
                    radial_range = None,
                    rawhdf: str = ""):
        
        # 生データをhdfファイルとして保存しているかどうかを確認する。
        # hdfファイルが読み込めるかどうかを確認する
        if rawhdf == "":
            flag_readhdf = False
        else:
            flag_readhdf = True
            with h5py.File(rawhdf, moder = "r") as f:
                if self.log:
                    h5_tree(f)
        
        integratedhdf = os.path.join(self.cachedir, self.name + "_integrate2d.hdf")
        with h5py.File(integratedhdf, mode = "w") as f:
            f.create_dataset(
                name = "rad",
                shape = (npt_rad,),
                dtype = np.float32,
            )
            f.create_dataset(
                name = "azim",
                shape = (npt_azim,),
                dtype = np.float32,
            )
            g = f.create_group(
                name = "integrated"
            )
            for i in (tqdm(range(len(self.filelist))) if self.log else range(len(self.filelist))):
                g.create_dataset(
                    name = "frame = {}".format(i),
                    shape = (npt_azim,npt_rad),
                    dtype = np.float32
                )
        
        ai = pyFAI.load(poni)
        lock = threading.Lock()

        def integrate2d(i):

            # データを読み込む
            if flag_readhdf:
                with lock:
                    with h5py.File(rawhdf, mode = "r") as f:
                        d = f["frame = {}".format(i)][()] # type: ignore
            else:
                d = his2array(self.filelist[i])

            # unrollする
            intensity, radial, azimuthal = ai.integrate2d(
                data = d,
                npt_rad = npt_rad,
                npt_azim = npt_azim,
                radial_range=radial_range,
                unit = "2th_deg",
                method = "ocl"
            )
            return i, intensity, radial, azimuthal

        # 平行作業を行う
        with confu.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpe:

            # 指示出し
            futures = [tpe.submit(integrate2d, i) for i in range(len(self.filelist))]

            # データ回収
            for i, future in enumerate(confu.as_completed(futures)):
                j, intensity, radial, azimuthal = future.result()
                with lock:
                    with h5py.File(integratedhdf, mode = "r+") as f:
                        if not i:
                            f["rad"][:] = radial # type: ignore
                            f["azim"][:] = azimuthal # type: ignore
                        f["integrated/frame = {}".format(j)][:] = intensity # type: ignore
                if self.log:
                    simple_progress_bar(i+1, len(self.filelist))

        # 出力
        if self.log:
            print("\nIntegration completed.")
            with h5py.File(integratedhdf, mode = "r") as f:
                h5_tree(f)       

        return integratedhdf    
        
    def integrate1D(self,
                    poni: str,
                    npt_rad: int,
                    radial_range = None,
                    rawhdf: str = ""):
        
        # 生データをhdfファイルとして保存しているかどうかを確認する。
        # hdfファイルが読み込めるかどうかを確認する
        if rawhdf == "":
            flag_readhdf = False
        else:
            flag_readhdf = True
            with h5py.File(rawhdf, moder = "r") as f:
                if self.log:
                    h5_tree(f)
        
        integratedhdf = os.path.join(self.cachedir, self.name + "_integrate1d.hdf")
        with h5py.File(integratedhdf, mode = "w") as f:
            f.create_dataset(
                name = "rad",
                shape = (npt_rad,),
                dtype = np.float32,
            )
            g = f.create_group(
                name = "integrated"
            )
            for i in (tqdm(range(len(self.filelist))) if self.log else range(len(self.filelist))):
                g.create_dataset(
                    name = "frame = {}".format(i),
                    shape = (npt_rad,),
                    dtype = np.float32
                )
        
        ai = pyFAI.load(poni)
        lock = threading.Lock()

        def integrate2d(i):

            # データを読み込む
            if flag_readhdf:
                with lock:
                    with h5py.File(rawhdf, mode = "r") as f:
                        d = f["frame = {}".format(i)][()] # type: ignore
            else:
                d = his2array(self.filelist[i])

            # 積算する
            radial, intensity = ai.integrate1d(
                data = d,
                npt = npt_rad,
                radial_range=radial_range,
                unit = "2th_deg",
                method = "ocl"
            )
            return i, intensity, radial

        # 平行作業を行う
        with confu.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpe:

            # 指示出し
            futures = [tpe.submit(integrate2d, i) for i in range(len(self.filelist))]

            # データ回収
            for i, future in enumerate(confu.as_completed(futures)):
                j, intensity, radial = future.result()
                with lock:
                    with h5py.File(integratedhdf, mode = "r+") as f:
                        if not i:
                            f["rad"][:] = radial # type: ignore
                        f["integrated/frame = {}".format(j)][:] = intensity # type: ignore
                if self.log:
                    simple_progress_bar(i+1, len(self.filelist))

        # 出力
        if self.log:
            print("\nIntegration completed.")
            with h5py.File(integratedhdf, mode = "r") as f:
                h5_tree(f)       

        return integratedhdf
