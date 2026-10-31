# 更新履歴
# 2026-10-07 作成

# 基本モジュール
import sys, os
sys.dont_write_bytecode = True
import time
from IPython.display import display, HTML, clear_output, update_display, Image
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import pyFAI
import pandas as pd
import h5py
from tqdm import tqdm
import PIL.Image as im
import json
from pprint import pprint
from typing import Optional, Union
# import pybeads

# fpdファイル取扱い用
import threading
import concurrent.futures as confu
import logging
logging.getLogger("pyFAI").setLevel(logging.ERROR)

# 自作モジュール
sys.path.append(r"J:\python\modules-main")
from Mytools.Tools import h5_tree, dict_tree, his2array, simple_progress_bar # type: ignore
import Mytools.Settings # type: ignore
sys.path.append(os.getcwd())

class create_hdfdata:
    def __init__(self):
        self.setting_hisfiles: dict = {
            'directory': "",
            'header': "",
            'footer': ".his"
        }
        self.filelist: list = []

        self.setting_hdffile: dict = {
            'filename': "output.hdf",
            'poni': "",
            'step': 0.005,
            'radial_range': (3,30)
        }

        self.setting_tiff: dict = {
            'filename': "output.tif",
            'frame_range': (0,1)
        }

        self.setting_csv: dict = {
            'filename': "output.csv",
            'frame_range': (0,1),
            'poni': "",
            'step': 0.005,
            'radial_range': (3,30),
            'beads': { # beadsの引数
                'fc': 0.005,
                'lam0': 1,
                'lam1': 5,
                'lam2': 4,
                'Nit': 20,
                'eps': 1e-6
            }
        }

    def set_filelist(self): 

        # 出力
        print("*"*40)
        print("directory:" + os.path.abspath(self.setting_hisfiles["directory"])) # type: ignore
        print("header:   " + self.setting_hisfiles["header"]) # type: ignore
        print("footer:   " + self.setting_hisfiles["footer"]) # type: ignore
        print("*"*40)

        # ファイルリストを作成し、ソートする
        for __ in os.listdir(self.setting_hisfiles["directory"]):
            if not self.setting_hisfiles["header"] in __: # type: ignore
                continue
            if not self.setting_hisfiles["footer"] in __: # type: ignore
                continue
            self.filelist.append(__)
        ## ソート
        self.filelist.sort(key = (lambda x: int(x.replace(self.setting_hisfiles["header"], "").replace(self.setting_hisfiles["footer"], ""))))
        self.filelist = list(map(lambda x: os.path.join(self.setting_hisfiles["directory"], x), self.filelist)) # type: ignore

    def initialize_hdf(self):

        # 出力
        print("*"*40)
        print("filename:    " + os.path.abspath(self.setting_hdffile["filename"])) # type: ignore
        print("poni:        " + self.setting_hdffile["poni"]) # type: ignore
        print("step:        {}".format(self.setting_hdffile["step"])) # type: ignore
        print("radial_range:{}".format(self.setting_hdffile["radial_range"])) # type: ignore
        print("*"*40)
        
        # 切り開き点数を計算する
        npt_rad = npt_rad = int((
            self.setting_hdffile["radial_range"][1] - self.setting_hdffile["radial_range"][0]
            ) / self.setting_hdffile["step"])

        with h5py.File(self.setting_hdffile["filename"], mode = "w") as f:
            f.create_dataset(
                name = "rad",
                shape = (npt_rad,),
                dtype = np.float32,
            )
            g = f.create_group(
                name = "integrated"
            )
            g.attrs["poni"] = self.setting_hdffile["poni"]
            g.attrs["length"] = len(self.filelist)
            g.attrs["step"] = self.setting_hdffile["step"]
            g.attrs["radial_range"] = self.setting_hdffile["radial_range"]
            for i in tqdm(range(len(self.filelist))):
                g.create_dataset(
                    name = "frame = {}".format(i),
                    shape = (npt_rad,),
                    dtype = np.float32
                )
        print("Make hdf file: " + os.path.abspath(self.setting_hdffile["filename"]))
        return

    def ppe_pyFAI(self):
        
        lock = threading.Lock() # multithreaddaderhtitlum用

        # データ読み取り
        with h5py.File(self.setting_hdffile["filename"], mode = "r") as f:
            poni: str = f["integrated"].attrs["poni"] # type: ignore
            length: int = f["integrated"].attrs["length"] # type: ignore
            step: int = f["integrated"].attrs["step"] # type: ignore
            radial_range: tuple = f["integrated"].attrs["radial_range"] # type: ignore
        npt_rad = npt_rad = int((radial_range[1] - radial_range[0])/step)

        print("*"*40)
        print("poni:         " + os.path.abspath(poni))
        print("length:       {}".format(length))
        print("step:         {}".format(step))
        print("radial_range: ({}, {})".format(*radial_range))
        print("*"*40)

        # 引数作成
        args = []
        for i in range(length):
            args.append({
                "index": i,
                "file": self.filelist[i],
                "npt_rad": npt_rad,
                "poni": poni,
                "radial_range": radial_range
            })

        # 切り開きを実行する
        with confu.ThreadPoolExecutor(max_workers=os.cpu_count()) as tpe:

            # ジョブを引き渡す
            futures = [tpe.submit(_func, args[i]) for i in range(length)]

            # ジョブが完了した順に実行
            for i, future in enumerate(confu.as_completed(futures)):
                j, intensity, radial = future.result()
                with lock:
                    # hdfファイルに書き込み
                    with h5py.File(self.setting_hdffile["filename"], mode = "r+") as f:
                        if not i:
                            f["rad"][:] = radial # type: ignore
                        f["integrated/frame = {}".format(j)][:] = intensity # type: ignore
                    simple_progress_bar(i+1, len(self.filelist))
        print("Complete!")

    def save_tiff(self):

        # 引数読み込み
        imgfilename = os.path.abspath(self.setting_tiff["filename"])
        frame_range = self.setting_tiff["frame_range"]
        
        # 引数の出力
        print("=" * 30)
        print("Input:")
        print("\tframe_range: {}-{}".format(*frame_range))
        print("\timgfilename: "+imgfilename)
        print("=" * 30)

        # imgファイルの生成
        hisdata = []
        for j in range(*frame_range):
            f = self.filelist[j]
            hisdata.append(his2array(f))
        img = np.average(np.stack(hisdata), axis = 0)

        ## 保存
        im.fromarray(img).save(imgfilename)
        print("[Save tiff]: " + imgfilename)

    def save_csv(self):

        # 引数読み込み
        csvfilename = os.path.abspath(self.setting_csv["filename"])
        frame_range = self.setting_csv["frame_range"]
        poni = self.setting_csv["poni"]
        step = self.setting_csv["step"]
        radial_range = self.setting_csv["radial_range"]
        pad_len = self.setting_csv["pad_len"]
        setting_beads = self.setting_csv["beads"]

        # 切り開き点数を計算する
        npt_rad = npt_rad = int((
            radial_range[1] - radial_range[0]
            ) / step)
        
        # 引数の出力
        print("=" * 30)
        print("Input:")
        pprint(self.setting_csv)
        print("=" * 30)

        # imgファイルの生成
        hisdata = []
        for j in range(*frame_range):
            f = self.filelist[j]
            hisdata.append(his2array(f))
        img = np.average(np.stack(hisdata), axis = 0)


        # インスタント化
        ai = pyFAI.load(poni)

        # 積算する
        radial, intensity = ai.integrate1d(
            data = img,
            npt = npt_rad,
            radial_range=radial_range,
            unit = "2th_deg",
            method = "ocl",
        )

        # beads補正する
        bd = pybeads()
        intensity_normalized = intensity-np.linspace(intensity[0], intensity[-1], intensity.shape[0])
        bd.setting = setting_beads
        signal_est, bg_est = bd.beads(
            intensity_normalized,
        )

        ## 保存
        df = pd.DataFrame(np.vstack([radial,signal_est,bg_est]).T) # type: ignore
        df.info()
        df.to_csv(csvfilename,header = False, index = False)
        print("Save: " + csvfilename)

def _func(arg: dict):
    
    # データ読み込み
    i = arg["index"]
    file = arg["file"]
    npt_rad = arg["npt_rad"]
    poni = arg["poni"]
    radial_range = tuple(arg["radial_range"])

    # データ読み込み
    d = his2array(file)

    # インスタント化
    ai = pyFAI.load(poni)

    # 積算する
    radial, intensity = ai.integrate1d(
        data = d,
        npt = npt_rad,
        radial_range=radial_range,
        unit = "2th_deg",
        method = "ocl",
    )

    return i, intensity, radial

class pybeads:

    def __init__(self):
        self.setting: dict = {
            'fc': 0.005,
            'lam0': 1,
            'lam1': 5,
            'lam2': 4,
            'Nit': 20,
            'eps': 1e-6
        }

    def _diff_matrix(self,
                    n,
                    order=1,
                    ):
        """
        差分行列
        """
        if order == 1:
            e = np.ones(n)
            D = sp.sparse.diags([e, -e], [1, 0],
                        shape=(n-1, n))
        elif order == 2:
            e = np.ones(n)
            D = sp.sparse.diags([e, -2*e, e],
                        [0, 1, 2],
                        shape=(n-2, n))
        else:
            raise ValueError

        return D.tocsr()

    def _lowpass_background(
            self,
            x,
            fc,
            ):
        """
        Butterworth低域通過フィルタで
        ベースライン推定
        """
        b, a = sp.signal.butter(
            N=2,
            Wn=fc,
            btype="low"
        )
        return sp.signal.filtfilt(b, a, x)

    def beads(
        self,
        y: np.ndarray
    ):
        """
        Simplified BEADS

        Parameters
        ----------
        y : ndarray
            観測データ

        Returns
        -------
        signal_est
        background_est
        """

        N = len(y)

        D1 = self._diff_matrix(N, 1)
        D2 = self._diff_matrix(N, 2)

        x = np.maximum(y, 0) # type: ignore

        for _ in range(self.setting["Nit"]):

            w0 = 1.0 / np.sqrt(x**2 + self.setting["eps"]) # type: ignore

            dx1 = D1 @ x
            dx2 = D2 @ x

            w1 = 1.0 / np.sqrt(dx1**2 + self.setting["eps"]) # type: ignore
            w2 = 1.0 / np.sqrt(dx2**2 + self.setting["eps"]) # type: ignore

            W0 = sp.sparse.diags(w0) # type: ignore

            W1 = sp.sparse.diags(w1) # type: ignore
            W2 = sp.sparse.diags(w2) # type: ignore

            A = (
                sp.sparse.eye(N) # type: ignore
                + self.setting["lam0"] * W0
                + self.setting["lam1"] * (D1.T @ W1 @ D1)
                + self.setting["lam2"] * (D2.T @ W2 @ D2)
            )

            rhs = y

            x = sp.sparse.linalg.spsolve(A, rhs) # type: ignore

            # 正ピーク制約
            x = np.maximum(x, 0) # type: ignore

        background = self._lowpass_background(
            y - x,
            self.setting["fc"]
        )

        signal = y - background

        return signal, background

if False:
    # インスタンス化
    ch1 = create_hdfdata()

    # hisファイルを指定
    ch1.setting_hisfiles = {
        'directory': r"E:\LAMBDA\2026_10_1_BL10XU\Okazaki\UODE57\FPD",
        'header': "UODE57_14_",
        'footer': ".his"
    }
    ch1.set_filelist()
    pprint(ch1.filelist)

    # type = input("Do you want to continue? [y/n]\n")
    # if type != "y":
    #     sys.exit()

    # # hdfファイルを設定
    # ch1.setting_hdffile = {
    #     'filename': r"E:\LAMBDA\2026_10_1_BL10XU\Okazaki\Analyze\output.hdf",
    #     'poni': r"E:\LAMBDA\2026_10_1_BL10XU\CeO2\FPD\pyFAIcalib2_FPD_MgS400_EH2_20261007.poni",
    #     'step': 0.005,
    #     'radial_range': (8,23)
    # }
    # ch1.initialize_hdf()

    # type = input("Do you want to continue? [y/n]\n")
    # if type != "y":
    #     sys.exit()

    # # 一次元化実施
    # ch1.ppe_pyFAI()

    # type = input("Do you want to continue? [y/n]\n")
    # if type != "y":
    #     sys.exit()

    # ch1.setting_csv = {
    #     'filename': r"E:\LAMBDA\2026_10_1_BL10XU\Okazaki\Analyze\UODE57_0014_00.csv",
    #     'frame_range': (0,1),
    #     'poni': r"E:\LAMBDA\2026_10_1_BL10XU\CeO2\FPD\pyFAIcalib2_FPD_MgS400_EH2_20261007.poni",
    #     'step': 0.02,
    #     'radial_range': (8,23),
    #     'pad_len': 100,
    #     'beads': { # beadsの引数
    #         'fc': 0.005,
    #         'lam0': 15,
    #         'lam1': 150,
    #         'lam2': 120,
    #         'Nit': 20,
    #         'eps': 1e-6
    #     }
    # }
    # ch1.save_csv()

    # type = input("Do you want to continue? [y/n]\n")
    # if type != "y":
    #     sys.exit()

    # ch1.setting_tiff = {
    #     'filename': r"E:\LAMBDA\2026_10_1_BL10XU\Okazaki\Analyze\UODE57_0014_12.tif",
    #     'frame_range': (841,863)
    # }
    # ch1.save_tiff()

if __name__ == "__main__":

    directory = r"E:\LAMBDA\2026_10_1_BL10XU\Okazaki\Analyze"

    csvfiles = []
    for f in os.listdir(directory):
        if ".csv" in f:
            if not "_beads" in f:
                csvfiles.append(f)
                # break

    setting_beads = {
        'fc': 0.005,
        'lam0': 15,
        'lam1': 150,
        'lam2': 120,
        'Nit': 20,
        'eps': 1e-6
    }
    pb = pybeads()
    pb.setting = setting_beads

    for f in csvfiles:
        df = pd.read_csv(os.path.join(directory, f),
                         header = None,
                         index_col = None)
        tth = df[0].values[:-1]
        intensity = df[1].values[:-1]
        intensity_nor = intensity - np.linspace(intensity[0], intensity[-1], intensity.shape[0])
        signal_est, bg_est = pb.beads(intensity_nor)

        df_new = pd.DataFrame({
            'tth': tth,
            'intensity': signal_est,
            'background': bg_est
        })        

        # plt.plot(df_new["tth"], df_new["intensity"])
        # plt.plot(df_new["tth"], df_new["background"])
        # plt.show()
        df_new.to_csv(os.path.join(directory, os.path.splitext(f)[0]+"_beads.csv"), header = False, index = False)

        