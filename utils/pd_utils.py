from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.fftpack import fft,fftshift
import pandas as pd

import matplotlib.pyplot as plt

def get_psd_values(y_values, N, fs):
    f_values, psd_values = welch(y_values, fs)
    return f_values, psd_values

def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))

def get_fft_values(y_values, N, fs):  # N为采样点数，f_s是采样频率，返回f_values希望的频率区间, fft_values真实幅值
    f_values = np.linspace(0.0, fs / 2.0, N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, N, fs):
    autocorr_values = autocorr(y_values)
    x_values = np.array([1.0 * jj / fs for jj in range(0, N)])
    return x_values, autocorr_values


def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a


def butter_highpass_filter(data, cutOff, fs, order=3):
    b, a = butter_highpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def fft_peak_xy(data,N,fs,peak_num=5):
    f_values, fft_values = get_fft_values(data, N, fs)
    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    peak_save = fft_values[peaks].argsort()[::-1][:peak_num]
    temp_arr = f_values[peaks[peak_save]] + fft_values[peaks[peak_save]] ## 峰值前5的极值点横纵坐标之和
    fft_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant', constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return fft_peak_xy


def psd_peak_xy(data,N,fs,peak_num=5):
    p_values, psd_values = get_psd_values(data, N, fs)
    signal_min = np.nanpercentile(psd_values, 5)
    signal_max = np.nanpercentile(psd_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(psd_values)  # set minimum peak height
    peaks3, _ = find_peaks(psd_values, height=mph)
    peak_save = psd_values[peaks3].argsort()[::-1][:peak_num]
    temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
    psd_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant',
                              constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return psd_peak_xy


def auto_peak_xy(data,N,fs,peak_num=5):
    a_values, autocorr_values = get_autocorr_values(data, N, fs)
    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(autocorr_values)  # set minimum peak height
    peaks4, _ = find_peaks(autocorr_values, height=mph)
    peak_save = autocorr_values[peaks4].argsort()[::-1][:peak_num]
    temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
    autocorr_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant',
                                   constant_values=0)  ## (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return autocorr_peak_xy


import math
import numpy as np
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal


# 输入信号序列即可(list)
def envelope_extraction(signal):
    s = signal.astype(float)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0, ]  # 上包络的x序列
    u_y = [s[0], ]  # 上包络的y序列

    l_x = [0, ]  # 下包络的x序列
    l_y = [s[0], ]  # 下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)  # 上包络与原始数据切点x
    u_y.append(s[-1])  # 对应的值

    l_x.append(len(s) - 1)  # 下包络与原始数据切点x
    l_y.append(s[-1])  # 对应的值

    # u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]  # 边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]  # 边界值处理
    lower_envelope_y[-1] = l_y[-1]

    # 上包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            # 求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

            # 下包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            # 求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

            # 也可以使用三次样条进行拟合
    # u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # for k in range(0,len(s)):
    #   q_u[k] = u_p(k)
    #   q_l[k] = l_p(k)

    return upper_envelope_y, lower_envelope_y


def general_equation(first_x, first_y, second_x, second_y):
    # 斜截式 y = kx + b
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

def mAmp(data):  #平均幅值
    L = np.size(data, 0) #计算data的样本数量大小
    upper_envolope, low_envolope = envelope_extraction(data)
    mAmp = (upper_envolope-low_envolope)/L*1.0
    return mAmp

def sampEn(data, N, r): #多窗口样本熵
    L = len(data)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([data[i: i + N] for i in range(L - N)])
    xmj = np.array([data[i: i + N] for i in range(L - N + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    N += 1
    xm = np.array([data[i: i + N] for i in range(L - N + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    # Return SampEn
    return -np.log(A / B)

def fft_domain(data,N,fs,peak_num=5):  ##频域图主峰
    f_values, fft_values = get_fft_values(data, N, fs)
    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    index_peakmax = fft_values[peaks].argsort()[-1]
    # 主峰
    peak_main_X = f_values[peaks[index_peakmax]]
    peak_main_Y = fft_values[peaks[index_peakmax]]
    return peak_main_X, peak_main_Y

def DF(X, Y, Z, N, fs, peak_num=5): ##多轴主峰最大amplitude
    _, fft_peak_x_main = fft_domain(X,N,fs,peak_num=5)
    _, fft_peak_y_main = fft_domain(Y,N,fs,peak_num=5)
    _, fft_peak_z_main = fft_domain(Z,N,fs,peak_num=5)
    return max(fft_peak_x_main, fft_peak_y_main, fft_peak_z_main)

from scipy.integrate import simps

def PSDEnergy_XYZ(X, Y, Z, N, fs): ##多轴功率谱密度能量值的和
    X_p, X_psd = get_psd_values(X, N, fs)
    Y_p, Y_psd = get_psd_values(Y, N, fs)
    Z_p, Z_psd = get_psd_values(Z, N, fs)
    x_energy = simps(X_psd, X_p, dx=0.001)
    y_energy = simps(Y_psd, Y_p, dx=0.001)
    z_energy = simps(Z_psd, Z_p, dx=0.001)
    return x_energy+y_energy+z_energy

import scipy.integrate as integrate


def spectrumConcentration(X, Y, Z, N, fs): ##主峰频谱集中区能量比
    DF_x, _ = fft_domain(X, N, fs, peak_num=5)
    DF_y, _ = fft_domain(Y, N, fs, peak_num=5)
    DF_z, _ = fft_domain(Z, N, fs, peak_num=5)
    X_spectrumDistribution = integrate.quad(get_fft_values(X, N, fs), DF_x - 0.4, DF_x + 0.4)
    Y_spectrumDistribution = integrate.quad(get_fft_values(Y, N, fs), DF_y - 0.4, DF_y + 0.4)
    Z_spectrumDistribution = integrate.quad(get_fft_values(Z, N, fs), DF_z - 0.4, DF_z + 0.4)
    spectrumDistribution = X_spectrumDistribution + Y_spectrumDistribution + Z_spectrumDistribution
    return spectrumDistribution / PSDEnergy_XYZ(X, Y, Z, N, fs) *1.0

def pic_psd(data_x,data_y,data_z,data_a, windowsize, fs,pd_num,j):
    p_x, psd_x = get_psd_values(data_x, windowsize, fs)
    p_y, psd_y = get_psd_values(data_y, windowsize, fs)
    p_z, psd_z = get_psd_values(data_z, windowsize, fs)
    p_a, psd_a = get_psd_values(data_a, windowsize, fs)
    plt.plot(p_x, psd_x, linestyle='-', color='green')
    plt.plot(p_y, psd_y, linestyle='-', color='red')
    plt.plot(p_z, psd_z, linestyle='-', color='blue')
    plt.plot(p_a, psd_a, linestyle='-', color='black')
    # plt.xlim(0, 25)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2 / Hz]')
    plt.title("PSD signals")
    plt.savefig("psd{}_window{}.png".format(pd_num,j+1))
    plt.show()
    plt.cla()
    
def pic_auto(data,windowsize,fs):
    a_values, autocorr_values = get_autocorr_values(data, windowsize, fs)
    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min)/len(autocorr_values)#set minimum peak height
    peaks4, _ = find_peaks(autocorr_values,height = mph) 
    peak_save = autocorr_values[peaks4].argsort()[::-1][:20]
    plt.plot(a_values, autocorr_values, linestyle='-', color='black')
    plt.scatter(a_values[peaks4[peak_save]], autocorr_values[peaks4[peak_save]],marker = "x", c='r')
    #plt.xlim(0,25)
    plt.xlabel('Time delay [s]')
    plt.ylabel('Amplitude')
    plt.title("Autocorrelation signals")

# lp_coefs = librosa.lpc(values, 3)   ##Linear prediction coefficients
# _, cD = pywt.dwt(values, ’db3’)  ##Wavelet transform detail coefficients (cD)
# third_order_cum = scipy.stats.moment(values, moment = 3)  ##Third order cumulant
#
# def powerRatio(data):   ##Power ratio (1–6 Hz/6–12 Hz)
#     xf = numpy.linspace(0, af / 2, frequency_values.size)
#     num = frequency_values[(xf > = 1) & (xf <= 6)]
#     den = frequency_values[(xf > = 6) & (xf <= 12)]
#     retun power_ratio = num.mean() / den.mean()

# def winTrendMean(data, N):  ##Mean Trend and Windowed Mean Difference
#     meanTrend = mea
#     winMeanDiff =
#     return meanTrend, winMeanDiff
#
# def winTrendVar(data, N):  ##Variance Trend and Windowed Variance Difference
#     varTrend =
#     winVarDiff =
#     return varTrend, winVarDiff
#
# def DFACoef():  ##Detrended fluctuation analysis(DFA) Coefficient
#     return