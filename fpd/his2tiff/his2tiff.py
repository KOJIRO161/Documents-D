import sys
sys.dont_write_bytecode = True # __pycache__を生まない

# 基本モジュール
import struct
import os
import numpy as np
from datetime import datetime
import cv2
import json

# tkinter
from tkinter import filedialog, messagebox, Tk
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1) # tkinterの解像度を上げる

# PerkinElmerの.hisファイル メタデータを読み込む関数
def read_his_metadata(filepath: str):

    if not os.path.exists(filepath):
        print("No such file.")
        return None

    metadata = dict()
    
    with open(filepath, 'rb') as f:
        # 先頭64バイトのヘッダーを読み込む
        header_data = f.read(64)
        
        if len(header_data) < 64:
            print("Invalid header size.")
            return None

        # struct.unpack を使用してバイナリデータを解釈
        # フォーマット文字列は一般的なPerkinElmerヘッダー(リトルエンディアン)を想定:
        # H: unsigned short (2 bytes), i: int (4 bytes)
        # 構造:
        # [0-1] File type (2 bytes)
        # [2-3] Len (2 bytes)
        # [4-7] Header size (4 bytes)
        # [8-9] Width (2 bytes)
        # [10-11] Height (2 bytes)
        # [12-13] Number of frames (2 bytes)
        
        file_type, length, header_size, width, height, num_frames = struct.unpack('<HHihhH', header_data[:14])
        
        metadata['FileType'] = file_type
        metadata['HeaderSize'] = header_size
        metadata['Width'] = width
        metadata['Height'] = height
        metadata['Frames'] = num_frames
        
    return metadata

# hisファイルを読み込んで、np.ndarrayにする関数
def his2array(filename: str):
    """
    :param filename: 読み込むファイル
    :type filename: str
    :return: 画像データ
    :retype: np.ndarray
    """
    with open(filename, 'rb') as file:
        data = file.read()
    
    keylist = [
        "ID", "headerSize", "headerVersion", "fileSize", "imageHeaderSize",
        "ULX", "ULY", "BRX", "BRY", "numberOfFrame", "correction",
        "frameTimeInMicroseconds", "frameTimeInMilliseconds"
    ]

    info = dict(
        zip(
            keylist,
            struct.unpack('<HHHLHHHHHHHdH', data[:34])
        )
    )

    # 【追加】デバッグ＆安全対策：ヘッダー情報がおかしい場合はエラーで知らせる
    if info.get("headerSize", 0) == 0 or info.get("numberOfFrame", 0) == 0:
        raise ValueError(f"【データ破損または形式違い】ファイル名: {filename}, ヘッダーサイズ: {info.get('headerSize')}")

    imageWidth = info["BRX"] - info["ULX"] + 1
    imageHeight = info["BRY"] - info["ULY"] + 1

    start_position = info["headerSize"] + info["imageHeaderSize"]

    image_data = np.frombuffer(data[start_position:], dtype=np.uint16)
    sequential_image_intensities = image_data.reshape((info["numberOfFrame"], imageHeight, imageWidth))

    img = sequential_image_intensities[0]
    return img

# 保存先の指定
def ask_savefilename(filetypes: list,
                     initialdir = None,
                     initfilename = None,
                     defaultextension = None,):
    """
    ask_savefilename の Docstring
    
    :param data: 保存するデータ
    :param filetypes: データの拡張子
    :type filetypes: list
    :param initialdir: 最初に表示するディレクトリ
    :param initfilename: ファイル名の初期値（Noneの場合、時間になる）
    :param defaultextension: デフォルト拡張子
    :return: ファイル名
    :retype: str
    """
    # 変更履歴
    ## 2025年12月16日作成

    # 現在時刻の取得
    if initfilename == None:
        dt = datetime.now()
        extension = filetypes[0][1][1:]
        initfilename = dt.strftime('%Y%m%d%H%M%S%f') + extension

    # tkinterでファイルダイアログを開く
    window = Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    filename = filedialog.asksaveasfilename(
        parent = window,
        filetypes = filetypes,
        initialfile=initfilename,
        initialdir=initialdir,
        defaultextension=defaultextension,
    )
    return filename

# 読込先の指定
def ask_openfilename(filetypes: list,
                     initialdir = None,):
    """
    ask_openfilename の Docstring
    
    :param filetypes: データの拡張子
    :type filetypes: list
    :param initialdir: 最初に表示するディレクトリ
    :param defaultextension: デフォルト拡張子
    :return: ファイル名
    :retype: str
    """
    # 変更履歴
    ## 2025年12月16日 作成

    # tkinterでファイルダイアログを開く
    window = Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    filename = filedialog.askopenfilename(
        parent = window,
        filetypes = filetypes,
        initialdir=initialdir,
    )
    return filename

# 実行例
if __name__ == "__main__":
    # ファイルの指定
    hisfile = ask_openfilename(filetypes = [("HIS", ".his"),])

    # データの読み込み
    metadata = read_his_metadata(hisfile)
    img = his2array(hisfile)

    # データの保存
    saveimg = ask_savefilename(filetypes = [("TIFF", ".tif"),], defaultextension = ".tif")
    cv2.imwrite(saveimg, img)
    with open(os.path.splitext(saveimg)[0] + ".json", mode = "w") as f:
        json.dump(metadata, f, indent = 4)
    