"""変更履歴
*2026-02-06 作成
"""

import sys
sys.dont_write_bytecode = True # cacheを出さない

import numpy as np
import scipy as sp

class peakfit:

    def __init__(self):

        self.initparams = dict()

        self.x = np.array([])

        self.y = np.array([])

        self.func_list = [
            "gaussian",
            "lorenzian",
            "psdVoigt",
            "asympsdVoigt",
        ]

    def background(self,
                   x: np.ndarray,
                   params: dict,
                   ) -> np.ndarray:
        
        b0 = params["b0"]
        b1 = params["b1"]

        value = b0 + b1*x
        return value

    def gaussian(self,
                 x: np.ndarray,
                 params: dict,
                 ) -> np.ndarray:
        """
        :param x: 軸の値
        :type x: np.ndarray
        :param params: {
            "center": ピーク位置 (float),
            "height": ピークの高さ (float, >0),
            "fwhm": 半値幅 (float, >0)
        }
        :type param: dict
        :return: 密度関数
        :rtype: np.ndarray
        """

        # 引数
        center = params["center"]
        height = params["height"]
        fwhm = params["fwhm"]
        
        # 係数
        sigma = fwhm/2/np.sqrt(2*np.log(2))
        value = height
        
        # 関数
        value *= np.exp(-np.square((x-center)/sigma)/2)

        return value
    
    def gaussian_area(self,
                      params: dict,
                      ) -> float:
        """
        :param params: {
            "height": ピークの高さ (float, >0),
            "fwhm": 半値幅 (float, >0),
        }
        :type params: dict
        :return: ガウシアンピークの面積
        :rtype: float
        """

        # 引数
        height = params["height"]
        fwhm = params["fwhm"]

        # 演算
        sigma = fwhm/2/np.sqrt(2*np.log(2))
        value = height * sigma * np.sqrt(2*np.pi)

        return value
    
    def lorenzian(self,
                  x: np.ndarray,
                  params: dict,
                  ) -> np.ndarray:
        """
        :param x: 軸の値
        :type x: np.ndarray
        :param params: {
            "center": ピーク位置 (float),
            "height": ピークの高さ (float, >0)
            "fwhm": 半値幅 (float, >0)
        }
        :type params: dict
        :return: 密度関数
        :rtype: np.ndarray
        """

        # 引数
        center = params["center"]
        height = params["height"]
        fwhm = params["fwhm"]
        
        # 係数
        hwhm = fwhm/2
        area = np.pi * height * hwhm
        value = area/np.pi

        # 関数
        value *= hwhm / (np.square(x - center) + np.square(hwhm))

        return value
    
    def lorenzian_area(self,
                       params: dict,
                       ) -> float:
        """
        :param params: {
            "height": ピークの高さ (float, >0),
            "fwhm": 半値幅 (float, >0),
        }
        :type params: dict
        :return: ローレンチアンピークの面積
        :rtype: float
        """

        # 引数
        height = params["height"]
        fwhm = params["fwhm"]

        # 演算
        hwhm = fwhm / 2
        value = np.pi * height * hwhm

        return value
    
    def psdVoigt(self,
                 x: np.ndarray,
                 params: dict,
                 ) -> np.ndarray:
        """
        :param x: 軸の値
        :type x: np.ndarray
        :param params: {
            "center": ピーク位置 (float),
            "height": ピークの高さ (float, >0),
            "fwhm": 半値幅 (float, >0),
            "factor": ローレンチアン比率 (float, 0~1),
        }
        :type params: dict
        :return: 密度関数
        :rtype: np.ndarray
        """

        # 引数
        factor = params["factor"]
        
        # ガウシアン
        g = self.gaussian(
            x = x,
            params = params,
        )
        
        # ローレンチアン
        l = self.lorenzian(
            x = x,
            params = params,
        )

        # 畳み込み
        value = factor*l + (1-factor)*g

        return value
    
    def psdVoigt_area(self,
                      params: dict,
                      ) -> float:
        """
        :param params: {
            "height": ピークの高さ (float, >0),
            "fwhm": 半値幅 (float, >0),
            "factor": ローレンチアン比率 (float, 0~1),
        }
        :type params: dict
        :return: ピークの面積
        :rtype: float
        """

        # 引数
        factor = params["factor"]
        
        # ガウシアン
        g = self.gaussian_area(params = params)
        
        # ローレンチアン
        l = self.lorenzian_area(params = params)

        # 畳み込み
        value = factor*l + (1-factor)*g

        return value
    
    def asympsdVoigt(self,
                     x: np.ndarray,
                     params: dict,
                     ) -> np.ndarray:
        """
        :param x: 軸の値
        :type x: np.ndarray
        :param params: {
            "center": ピーク位置 (float),
            "height": ピークの高さ (float, >0),
            "left": {
                "fwhm": 半値幅 (float, >0),
                "factor": ローレンチアン比率 (float, 0~1),
            },
            "right": {
                "fwhm": 半値幅 (float, >0),
                "factor": ローレンチアン比率 (float, 0~1),
            },
        }
        :type params: dict
        :return: 密度関数
        :rtype: np.ndaary
        """

        # 引数
        center = params["center"]
        height = params["height"]

        # for文用のパラメーター
        masks = {
            'left': (x < center),
            'right': (x >= center),
        }
        value = np.zeros(x.shape)

        # 関数
        for k in masks.keys():
            
            # 引数
            params_tmp = params[k].copy()
            params_tmp["center"] = center
            params_tmp["height"] = 1

            # 演算
            value += self.psdVoigt(
                x = x,
                params = params_tmp,
            )*(masks[k].astype(np.float64))
        
        # 高さ補正
        value *= height
        
        return value
    
    def asympsdVoigt_area(self,
                          params: dict,
                          ) -> float:
        """
        :param params: {
            "height": ピークの高さ (float, >0),
            "left": {
                "fwhm": 半値幅 (float, >0),
                "factor": ローレンチアン比率 (float, 0~1),
            },
            "right": {
                "fwhm": 半値幅 (float, >0),
                "factor": ローレンチアン比率 (float, 0~1),
            },
        }
        :type params: dict
        :return: ピーク面積
        :rtype: float
        """
        
        # 格納用変数の定義
        value = 0

        for side in ["left", "right"]:
            params_tmp = params[side].copy()
            params_tmp["height"] = params["height"]
            value += self.psdVoigt_area(params_tmp)/2

        return value    
    
    def dict_to_lists(self,
                      d: dict,
                      ) -> tuple[list, list]:
        """
        辞書(d)からキーのリストと値のリストに変換する。
        ネスト構造に対応するため、キーはタプルとして階層構造を保持する。
        :params d: dictionary
        :type d: dict
        :return: (keys, values)
        :rtype: tuple[list, list]
        """
        keys = []
        values = []

        def _traverse(current_d, current_path):
            for k, v in current_d.items():
                new_path = current_path + (k,)
                if isinstance(v, dict):
                    # 値が辞書ならさらに深く探索（再帰）
                    _traverse(v, new_path)
                else:
                    # 辞書でなければ末端の値として保存
                    keys.append(new_path)
                    values.append(float(v)) # floatとして保存

        _traverse(d, ())
        return keys, values

    def lists_to_dict(self,
                      keys: list,
                      values: list,
                      ) -> dict:
        """
        キーのリストと値のリストからネストされた辞書(d)に変換する。
        キーのタプル構造に基づいて階層を復元する。
        :params keys: 辞書のkey
        :type keys: list
        :params values: 辞書の内容
        :type values: list
        :return: 辞書
        :rtype: dict
        """
        d = {}
        for key_path, val in zip(keys, values):
            current_level = d
            # 最下層のキーの1つ手前まで辞書を掘り下げる（なければ作成）
            for k in key_path[:-1]:
                current_level = current_level.setdefault(k, {})
            
            # 最下層に値をセット
            current_level[key_path[-1]] = val
            
        return d
    
    def calc_initparams(self,
                        **kargs,
                        ) -> dict:
        """
        初期値を推定する

        :param x: Value of x
        :type x: np.ndarray
        :param y: Value of y
        :type y: np.ndarray
        :param func: Fitting function
        :type func: str
        :return: Initial parameters for fitting
        :rtype: dict
        """

        # 引数
        x: np.ndarray = kargs.get("x", self.x.copy())
        y: np.ndarray = kargs.get("y", self.y.copy())
        func_label: str = kargs.get("func", "asympsdVoigt")

        # func_labelの確認
        if not func_label in self.func_list:
            raise KeyError("{} is not in the function list. Please provide valid function, such as {}.".format(func_label, self.func_list))

        # 格納用変数の定義
        value = dict()

        # backgroundの推定 (y = b0 + b1*x)
        b1 = (y[0] - y[-1])/(x[0] - x[-1])
        b0 = y[0] - b1*x[0]
        value["b0"] = b0
        value["b1"] = b1

        # centerの推定
        center = np.mean(x)
        value["center"] = center

        # ピーク高さの推定
        height = np.max(y) - (b0 + b1*center)
        value["height"] = height

        # 半値幅の推定
        if func_label == "asympsdVoigt":
            value["left"] = dict()
            value["right"] = dict()
            for k in ["left", "right"]:
                value[k]["fwhm"] = (x[-1] - x[0])/4
        else:
            value["fwhm"] = (x[-1] - x[0])/4

        # ローレンチアン比率の推定
        if func_label == "psdVoigt":
            value["factor"] = 0.5
        elif func_label == "asympsdVoigt":
            for k in ["left", "right"]:
                value[k]["factor"] = 0.5
        else:
            pass

        return value

    def fit(self,
            **kargs,
            ) -> dict:
        """
        フィッティングを行う

        :param x: Value of x (optional)
        :type x: np.ndarray
        :param y: Valume of y (optional)
        :type y: np.ndarray
        :param initparams: initial parameters for fitting (optional)
        :type initparams: dict
        :param func: fitting function
        :type func: str
        :return: {
            "params": fitting result,
            "err": fitting err,
            "popt": p optimize,
            "pcov": p covariance matrix,
            "key": params' key for optimizing,
            "func": fitting function
        }
        :rtype: dict
        """
        
        x = kargs.get("x", self.x)
        y = kargs.get("y", self.x)
        ip = kargs.get("initparams", self.initparams)
        func_label = kargs.get("func", "asympsdVoigt")

        # 初期値のkeyとvalueを取得
        ip_keys, ip_values = self.dict_to_lists(ip)

        # それぞれの関数のbounds, funcを定義
        bounds_up = [np.inf, np.inf]
        bounds_down = [-np.inf, -np.inf]
        if func_label == "gaussian": 
            bounds_up += [x[-1], np.inf, np.inf]
            bounds_down += [x[0], 0, 0]
            f = self.gaussian
        elif func_label == "lorenzian": 
            bounds_up += [x[-1], np.inf, np.inf]
            bounds_down += [x[0], 0, 0]
            f = self.lorenzian
        elif func_label == "psdVoigt":
            bounds_up += [x[-1], np.inf, np.inf, 1]
            bounds_down += [x[0], 0, 0, 0]
            f = self.psdVoigt
        elif func_label == "asympsdVoigt":
            bounds_up += [x[-1], np.inf, np.inf, 1, np.inf, 1]
            bounds_down += [x[0], 0, 0, 0, 0, 0]
            f = self.asympsdVoigt
        bounds = (
            tuple(bounds_down),
            tuple(bounds_up)
        )

        # フィッティング用の関数を定義
        def func(x, *args):
            params = self.lists_to_dict(ip_keys, list(args))
            return self.background(x, params) + f(x, params)
        
        # フィッティングを行う
        methods = ["trf", "dogbox"]
        for method in methods:
            try:
                popt, pcov = sp.optimize.curve_fit(
                    func,
                    x,
                    y,
                    p0 = ip_values,
                    maxfev = 4000,
                    bounds = bounds,
                    method = method,
                )
            except RuntimeError as errorcontent:
                if method == methods[-1]:
                    return errorcontent # type: ignore
                pass

        # フィッティング結果を辞書に直す
        resparams = self.lists_to_dict(ip_keys, popt)
        res = {
            "params": resparams,
            "err": self.lists_to_dict(keys = ip_keys, values = list(np.sqrt(np.diag(pcov)))),
            "popt": popt,
            "pcov": pcov,
            "key": ip_keys,
            "func": (lambda x, params: (self.background(x, params) + f(x, params)))
        }
        
        return res
