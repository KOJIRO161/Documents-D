import numpy as np
import scipy as sp
# import scipy.sparse as sp
# import scipy.sparse.linalg as spla
# from scipy.signal import butter, filtfilt


def diff_matrix(n, order=1):
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


def lowpass_background(x, fc):
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
    y,
    fc=0.005,
    lam0=1.0,
    lam1=5.0,
    lam2=4.0,
    Nit=20,
    eps=1e-6
):
    """
    Simplified BEADS

    Parameters
    ----------
    y : ndarray
        観測データ
    fc : float
        baseline cutoff
    lam0, lam1, lam2 : float
        BEADS正則化係数
    Nit : int
        反復回数

    Returns
    -------
    signal_est
    background_est
    """

    y = np.asarray(y, dtype=float)

    N = len(y)

    D1 = diff_matrix(N, 1)
    D2 = diff_matrix(N, 2)

    x = np.maximum(y, 0)

    for _ in range(Nit):

        w0 = 1.0 / np.sqrt(x**2 + eps)

        dx1 = D1 @ x
        dx2 = D2 @ x

        w1 = 1.0 / np.sqrt(dx1**2 + eps)
        w2 = 1.0 / np.sqrt(dx2**2 + eps)

        W0 = sp.sparse.diags(w0)

        W1 = sp.sparse.diags(w1)
        W2 = sp.sparse.diags(w2)

        A = (
            sp.sparse.eye(N)
            + lam0 * W0
            + lam1 * (D1.T @ W1 @ D1)
            + lam2 * (D2.T @ W2 @ D2)
        )

        rhs = y

        x = sp.sparse.linalg.spla.spsolve(A, rhs)

        # 正ピーク制約
        x = np.maximum(x, 0)

    background = lowpass_background(
        y - x,
        fc
    )

    signal = y - background

    return signal, background

