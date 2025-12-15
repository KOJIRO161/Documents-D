"""環境"""
# .lab

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

# グラフ等作成用
import matplotlib
import matplotlib.pyplot as plt         # 図の作成用
from PIL import Image as im
# import cv2
from IPython.display import display, HTML, clear_output, update_display, Image

# 自作モジュール
sys.path.append(r"C:\Users\okaza\pythonenv")
from modules.Mytools.Tools import print_fileinfo, h5_tree, dict_tree, simple_progress_bar, clean_cache_except_logfiles, get_total_size
from modules.Mytools.handle_ipynb import save_pickle, load_pickle, export_html
import modules.Mytools.Settings
import modules.fitXRD as fx
from modules.peakfit import peakfit, pseudoVoigt

# his読み込み用
import threading
import concurrent.futures as confu
from modules.Mytools.Tools import his2array


class Series2img:
    def __init__(self,
                 clear_cachedir: bool = False,
                 cachedir: str= "",
                 log: bool = True):
        
        self.cachedir = os.path.join(os.getcwd(), ".cache") if (cachedir == "") else cachedir
        self.log = log
        self.version = "1.0"
        self.rawfile = ""
        self.cakingfile = ""
        self.n_frame = 0

        self.cachedir: str = os.path.join(os.getcwd(), ".cache")
        if clear_cachedir:
            clean_cache_except_logfiles(self.cachedir)
        if self.log:
            print("cachedir: " + self.cachedir)
            print("log: {}".format(self.log))
            print("version: {}".format(self.version))
        
        self._set_dict()

    def _set_dict(self):
        self._dict = {
            "cachedir": self.cachedir,
            "log": self.log,
            "version": self.version,
            "rawfile": self.rawfile,
            "cakingfile": self.cakingfile,
            "n_frame": self.n_frame,
        }

    @property
    def dict(self):
        self._set_dict()
        return self._dict

    def his2hdf(self,
                dir: str,
                header: str,
                footer: str,
                initfilename: str = ""):
        
        """
        ディレクトリ(dir)内でheaderとfooterを含むファイルを検索し、リストに加えます。
        また、headerとfooterを除いた文字列を数字に変換し、ソートを行います。
        --> self.rawfile, self.n_frame
        """
        
        # tkinterでファイルダイアログを開く
        window = Tk()
        window.wm_attributes("-topmost", 1)
        window.withdraw()
        key = sys._getframe().f_code.co_name
        self.rawfile = filedialog.asksaveasfilename(
            parent = window,
            filetypes = [
                ("hdf", "*.hdf"),
            ],
            initialfile=key if (initfilename == "") else initfilename,
            initialdir=self.cachedir,
            defaultextension=".hdf",
        )
        with h5py.File(self.rawfile, mode = "w"):
            pass

        # ヘッダーとフッターを含むファイル名を取得
        flist = list()
        for __ in os.listdir(dir):
            if not header in __:
                continue
            if not footer in __:
                continue
            flist.append(__)

        # ソート
        flist.sort(key = (lambda x: int(x.replace(header, "").replace(footer, ""))))
        self.n_frame = len(flist)

        # 表示
        if self.log:
            print("--- sorted file list ----")
            for f in flist:
                print(os.path.join(dir, f))
            print("-------------------------" + "\n")
            print("n_frame: {}".format(self.n_frame))

        # データを更新
        for i, f in enumerate(flist):
            d = his2array(os.path.join(dir, f))
            with h5py.File(self.rawfile, mode = "r+") as f:
                f.create_dataset(
                    name = "frame = {}".format(i),
                    data = d,
                    dtype = d.dtype,
                    shape = d.shape,
                )
            if self.log:
                simple_progress_bar(i+1, len(flist))

        if self.log:
            print()
            print("rawfile: " + os.path.abspath(self.rawfile))
            with h5py.File(self.rawfile, mode = "r") as f:
                h5_tree(f)
        
        self._set_dict()

    def caking(self,
               poni: str,
               rad: int,
               initfilename: str = ""):

        """"
        cakingを行います。
        --> self.cakingfile
        """

        self.poni = poni
        if self.log:
            print("poni: " + os.path.abspath(self.poni))

        # tkinterでファイルダイアログを開く
        window = Tk()
        window.wm_attributes("-topmost", 1)
        window.withdraw()
        key = sys._getframe().f_code.co_name
        self.cakingfile = filedialog.asksaveasfilename(
            parent = window,
            filetypes = [
                ("hdf", "*.hdf"),
            ],
            initialfile=key if (initfilename == "") else initfilename,
            initialdir=self.cachedir,
            defaultextension=".hdf",
        )

        # 出力ファイル初期化
        with h5py.File(self.cakingfile, mode = "w") as f:
            f.create_dataset(
                name = "tth",
                shape = (rad,),
                dtype = np.float64
            )
            g = f.create_group(
                name = "integrated"
            )
            for i in tqdm(range(self.n_frame)):
                g.create_dataset(
                    name = "frame = {}".format(i),
                    shape = (rad,),
                    dtype = np.float32
                )
        if self.log:
            print("cakingfile: " + self.cakingfile)

        ai = pyFAI.load(self.poni)
        lock = threading.Lock()
        if self.log:
            print("ponifile: " + self.poni)

        # 切り開き用関数の定義
        def integrate(i):
            with lock:
                with h5py.File(self.rawfile, mode = "r") as f:
                    d = f["frame = {}".format(i)][()] # type: ignore
            tth, insty = ai.integrate1d(
                d,
                npt = rad,
                unit = "2th_deg",
                method = "ocl"
            )
            return i, insty, tth
        if self.log:
            print("Defined a function")

        # 演算
        with confu.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpe:

            # 演算開始
            futures = [tpe.submit(integrate, i) for i in range(self.n_frame)]

            # 終わったプロセスから順に出力
            for i, future in enumerate(confu.as_completed(futures)):
                j, insty, tth = future.result()
                with lock:
                    with h5py.File(self.cakingfile, mode = "r+") as f:
                        if not i:
                            f["tth"][:] = tth # type: ignore
                        f["integrated/frame = {}".format(j)][:] = insty # type: ignore
                simple_progress_bar(i+1, self.n_frame)
            print()

        if self.log:
            with h5py.File(self.cakingfile, mode = "r") as f:
                h5_tree(f)

        self._set_dict()

    def imshow_1dseries(self,
                        fps: float,
                        theta_lim: tuple = (),
                        fit: bool = False):
        
        """
        画像出力を行います。
        return: image file path
        """
        
        # データ読み込み
        with h5py.File(self.cakingfile, mode = "r") as f:
            tth = np.array(f["tth"][()]) # type: ignore
        
        # maskを作成
        mask = np.ones(tth.shape).astype(np.bool_)
        if theta_lim:
            mask[tth < theta_lim[0]] = False
            mask[tth > theta_lim[1]] = False

        # データ作成
        series = []
        for i in range(self.n_frame):
            with h5py.File(self.cakingfile, mode = "r") as f:
                series.append(
                    np.array(f["integrated/frame = {}".format(i)][()])[mask] # type: ignore
                )
        series = np.vstack(series)

        # ピークフィット
        if self.log:
            print("fit: {}".format(fit))
        if not "fit" in locals():
            fit = False
        if fit:
            pf = peakfit()
            mus = []
            for s in tqdm(series):
                _res = pf.fit_Vigot_func(
                    theta = tth[mask],
                    intensity=s
                )
                mu = pf.variables().index("mu")
                mus.append(_res[0][mu])
            
        # figureを作成
        fig, ax = plt.subplots()
        size_inches = (6,4.5)
        dpi = 300
        fig.set_size_inches(size_inches)

        # plot
        tthstep = abs(tth[0]-tth[1])
        ax.imshow(
            series.T,
            aspect = "auto",
            extent = (
                -0.5/fps,
                (self.n_frame-0.5)/fps,
                tth[mask][-1] + tthstep/2,
                tth[mask][0] - tthstep/2
            ),
            cmap = "inferno",
        )
        ax.autoscale(tight = True)
        ax.set_ylabel("2Theta [deg]", fontsize = 14)
        ax.set_xlabel("Time [s]", fontsize = 14)

        if fit:
            ax.plot(
                np.arange(self.n_frame)/fps,
                mus,
                lw = 0,
                marker = "o",
                c = "tab:blue",
                ms = 3
            )

        # 画像の表示
        key = sys._getframe().f_code.co_name
        imgfilename = os.path.join(self.cachedir, key + ".png")
        fig.savefig(imgfilename, dpi = dpi, transparent = True)
        plt.close()
        return imgfilename

    def save_tiff(self,
                  f_range: tuple,
                  vmin: float,
                  vmax: float,
                  step: int,
                  initfilename: str = ""):
        
        """
        tiffとpngを保存します。
        return: image file path
        """
        
        with h5py.File(self.rawfile, mode = "r") as f:
            for i, j in enumerate(range(*f_range)):
                if not i:
                    image = np.zeros(
                        shape = f["frame = {}".format(j)].shape # type: ignore
                    ).astype(f["frame = {}".format(j)].dtype) # type: ignore
                image += f["frame = {}".format(j)][()] # type: ignore

        # tkinterでファイルダイアログを開く
        window = Tk()
        window.wm_attributes("-topmost", 1)
        window.withdraw()
        key = sys._getframe().f_code.co_name
        tiffile = filedialog.asksaveasfilename(
            parent = window,
            filetypes = [
                ("tiff", "*.tiff"),
            ],
            initialfile=key if (initfilename == "") else initfilename,
            initialdir=self.cachedir,
            defaultextension=".tif",
        )
        if not tiffile:
            return
        im.fromarray(image).save(tiffile)

        bins = np.linspace(vmin, vmax, step)
        hist, bins = np.histogram(image, bins = bins)

        fig = plt.figure()
        size_inches = (6,8)
        dpi = 300
        fig.set_size_inches(size_inches)

        ax_imshow = fig.add_axes(rect = (
            1/6,
            3/8,
            4/6,
            4/8
        ))
        ax_imshow.imshow(
            image,
            cmap = "gray_r",
            vmin = vmin,
            vmax = vmax,
        )

        ax_hist = fig.add_axes(rect = (
            1/6,
            1/8,
            4/6,
            1.5/8
        ))
        ax_hist.step(
            (bins[1:] + bins[:-1])/2,
            hist,
            where = "mid",
            c = "0"
        )
        ax_hist.set_xlim(bins[0], bins[-1])
        ax_hist.set_ylim(0, ax_hist.get_ylim()[1])

        key = sys._getframe().f_code.co_name
        imgfile = os.path.splitext(tiffile)[0] + ".png"
        plt.savefig(imgfile, transparent = True, dpi = dpi)
        plt.close()

        return imgfile
