from numpy import *
import cmath as cm
import numpy as np
from TMM_tools.core.constants import *


def Eta(n, theta, mode='TE'):
    theta = theta * pi / 180
    cos_theta = cos(theta)
    sqrt_e0_m0 = sqrt(epsilon_0 / mu_0)
    eta = sqrt_e0_m0 * n / cos_theta if mode == 'TM' else sqrt_e0_m0 * n * cos_theta
    return eta


def Theta_Reverse(n_in, n_out, theta_in):
    theta_in = theta_in * pi / 180
    theta_out = cm.asin(n_in * sin(theta_in) / n_out)
    theta_out = theta_out * 180 / pi
    return theta_out


def Delta(f, n, d, theta):
    theta = theta * pi / 180
    w = 2 * pi * f
    delta = w / c * n * d * cos(theta)
    d_r = np.real(delta)
    d_i = np.imag(delta)
    d_r = np.mod(d_r, 2 * pi)
    delta = d_r + 1j * d_i
    return delta


def Normal_material_matrix(delta, eta):
    cos_delta = cos(delta)
    sin_delta = sin(delta)
    M = np.array([[cos_delta, -1j * sin_delta / eta],
                  [-1j * sin_delta * eta, cos_delta]])
    return M


def TR_coefficient(Matrix, theta, mode="TE", upper_background_n=1, lower_background_n=1,coefficient=True):
    m11, m12, m21, m22 = Matrix[0, 0], Matrix[0, 1], Matrix[1, 0], Matrix[1, 1]
    eta_0 = Eta(upper_background_n, theta, mode=mode)
    eta_N1 = Eta(lower_background_n, theta, mode=mode)
    low=(m11 * eta_0 + m12 * eta_0 * eta_N1 + m21 + m22 * eta_N1)
    t = 2 * eta_0 / low
    r = (m11 * eta_0 + m12 * eta_0 * eta_N1 - m21 - m22 * eta_N1) / low
    if coefficient:
        return t,r
    else:
        T,R=abs(t) ** 2, abs(r) ** 2
        A=1-T-R
        return T,R,A


if __name__ == '__main__':
    ...
