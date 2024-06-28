#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:59 2023

@author: macpro
"""

import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fftpack
from scipy.signal import find_peaks
import numpy as np
from scipy.special import entr
from scipy.fftpack import fft, fftshift
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import *
from utils import tremor_utils
from scipy.signal import welch
import string
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from utils import pd_utils

# scaler = StandardScaler()a
# x_train = scaler.fit_transform(x_np)
# 定义一个函数，输入是一个dataframe，输出也是一行提出的特征，
def longShortTermFeature(x, y, z, ACCW2, windowsize, overlapping, frequency):
    
    N = windowsize
    fs = frequency
    f_s = frequency
    data = ACCW2
    i = 0
    j = 0
    
    xmean = x.mean()
    xvar = x.var()
    ymean = y.mean()
    yvar = y.var()
    zmean = z.mean()
    zvar = z.var()
    amean = ACCW2.mean()
    avar = ACCW2.var()

    # z_filter = tremor_utils.butter_bandpass_filter(z,0.2, 2, 200, order = 4)
    z_filter = np.array(z).flatten()
    signal_min = np.nanpercentile(z_filter,5)
    signal_max = np.nanpercentile(z_filter, 97)
    mph = signal_max + (signal_max - signal_min)/len(z_filter)#set minimum peak height
    peaks_t, _ = find_peaks(z_filter,prominence=mph,distance=120) 
    peak_num=len(peaks_t) ##z轴peak数量
    t_value = np.arange(len(z_filter))
    t_peakmax = np.argsort(z_filter[peaks_t])[-1]
    sampley=pd_utils.sampEn(t_value[peaks_t],3,500)#z轴peak y样本熵
    samplex=pd_utils.sampEn(z_filter[peaks_t],3,1)#z轴peak x样本熵
    infory=pd_utils.infor(t_value[peaks_t])#z轴peak x信息熵
    inforx=pd_utils.infor(z_filter[peaks_t])#z轴peak y信息熵
    
    # t_peakmax_X = t_value[peaks_t[t_peakmax]]
    t_peakmax_Y = z_filter[peaks_t[t_peakmax]]
    t_peak_y = z_filter[peaks_t]
    dyski_num=len(t_peak_y[(t_peak_y< t_peakmax_Y-mph)])##z轴异常peak数量
    
    # auto_X = a_values[peaks4[index_peakmax]]
    a_values, autocorr_values = pd_utils.get_autocorr_values(z_filter, N, fs)
    peaks4, _ = find_peaks(autocorr_values)
    auto_peak_num=len(peaks4)
    index_peakmax = np.argsort(autocorr_values[peaks4])[-1]
    auto_y = autocorr_values[peaks4[index_peakmax]] #全局自相关系数
    
    #whole
    peaks_normal = np.zeros([len(data), 1], dtype=float)
    fea_sampley = np.zeros([len(data), 1], dtype=float)
    fea_samplex = np.zeros([len(data), 1], dtype=float)
    fea_infory = np.zeros([len(data), 1], dtype=float)
    fea_inforx = np.zeros([len(data), 1], dtype=float)
    peaks_abnormal = np.zeros([len(data), 1], dtype=float)
    fea_autoy = np.zeros([len(data), 1], dtype=float)
    meandif = np.zeros([len(data), 1], dtype=float)
    vardif = np.zeros([len(data), 1], dtype=float)
    fea_auto_num = np.zeros([len(data), 1], dtype=float)
    time_domain1 = np.zeros([len(data), 14], dtype=float)
    time_domain2 = np.zeros([len(data), 14], dtype=float)
    time_domain3 = np.zeros([len(data), 14], dtype=float)
    time_domain4 = np.zeros([len(data), 14], dtype=float)
    time_axiscof = np.zeros([len(data), 6], dtype=float)
    fre_domain1 = np.zeros([len(data), 19], dtype=float)
    fre_domain2 = np.zeros([len(data), 19], dtype=float)
    fre_domain3 = np.zeros([len(data), 19], dtype=float)
    fre_domain4 = np.zeros([len(data), 19], dtype=float)
    fre_df = np.zeros(len(data), dtype=float)
    fft_peak_a = np.zeros([len(data), 5], dtype=float)
    psd_domain1 = np.zeros([len(data), 19], dtype=float)
    psd_domain2 = np.zeros([len(data), 19], dtype=float)
    psd_domain3 = np.zeros([len(data), 19], dtype=float)
    psd_domain4 = np.zeros([len(data), 19], dtype=float)
    PSDEnergy_XYZ = np.zeros(len(data), dtype=float)
    spectrumConcentration = np.zeros(len(data), dtype=float)
    psd_peak_a = np.zeros([len(data), 5], dtype=float)
    autoco_domain1 = np.zeros([len(data), 19], dtype=float)
    autoco_domain2 = np.zeros([len(data), 19], dtype=float)
    autoco_domain3 = np.zeros([len(data), 19], dtype=float)
    autoco_domain4 = np.zeros([len(data), 19], dtype=float)
    autocorr_peak_a = np.zeros([len(data), 5], dtype=float)
    # label
    # label1 = np.zeros(len(data), dtype=float)
    # label1[0:len(data) - windowsize] = label_pd[0:len(data) - windowsize]
    # label2 = np.zeros(len(data), dtype=float)
    # label2[0:len(data) - windowsize] = label_pd[0:len(data) - windowsize]
    print("len_data-windowsize", len(data) - windowsize)


    while (i < len(data) - windowsize):
        data1 = x[i:i + windowsize]
        data2 = y[i:i + windowsize]
        data3 = z[i:i + windowsize]
        data4 = ACCW2[i:i + windowsize]
        data1 = data1.values  # dataframe转numpy数组
        data2 = data2.values
        data3 = data3.values
        data4 = data4.values
        #***************************(与data4/window无关特征)*******************
        peaks_normal[j,:] = peak_num #z轴0.2-2滤波后的波峰个数
        peaks_abnormal[j,:] = dyski_num  #z轴异常波峰个数
        fea_sampley[j,:] = sampley
        fea_samplex[j,:] = samplex
        fea_infory[j,:] = infory
        fea_inforx[j,:] = inforx
        
        fea_autoy[j,:] = auto_y
        fea_auto_num[j,:]=auto_peak_num
        
        meandif[j,:] = meandif[j,:]+abs(data1.mean()-xmean)+abs(data2.mean()-ymean)+abs(data3.mean()-zmean)
        vardif[j,:] = vardif[j,:]+abs(data1.var()-xvar)+abs(data2.var()-yvar)+abs(data3.var()-zvar)
        
        #***************************************short term features******************************
        # smv
        time_domain1[j,:] = tremor_utils.time_domain(data1)#14
        time_domain2[j,:] = tremor_utils.time_domain(data2)
        time_domain3[j,:] = tremor_utils.time_domain(data3)
        time_domain4[j,:] = tremor_utils.time_domain(data4)
        time_axiscof[j, :] = tremor_utils.corrcoef(data1,data2,data3,data4)
        fre_domain1[j,:] = tremor_utils.fft_domain(data1, N, fs)#19
        fre_domain2[j,:] = tremor_utils.fft_domain(data2, N, fs)
        fre_domain3[j,:] = tremor_utils.fft_domain(data3, N, fs)
        fre_domain4[j,:] = tremor_utils.fft_domain(data4, N, fs)
        fre_df[j] = tremor_utils.DF(data1,data2,data3,N,fs)
        fft_peak_a[j, :] = tremor_utils.fft_peak_xy(data4, N, fs, peak_num=5)
        psd_domain1[j,:] = tremor_utils.psd_domain(data1, N, fs)#19
        psd_domain2[j,:] = tremor_utils.psd_domain(data2, N, fs)
        psd_domain3[j,:] = tremor_utils.psd_domain(data3, N, fs)
        psd_domain4[j,:] = tremor_utils.psd_domain(data4, N, fs)
        PSDEnergy_XYZ[j] = tremor_utils.PSDEnergy_XYZ(data1,data2,data3,N,fs)
        spectrumConcentration[j] = tremor_utils.spectrumConcentration(data1,data2,data3,N,fs)
        psd_peak_a[j,:]  = tremor_utils.psd_peak_xy(data4, N, fs, peak_num=5)
        data1234 = np.c_[data1,data2,data3,data4]
        data1234 = StandardScaler().fit_transform(data1234)#19
        data1 = data1234[:,0]
        data2 = data1234[:,1]
        data3 = data1234[:,2]
        data4 = data1234[:,3]
        # data1=data1.apply(lambda x : (x-np.mean(x))/(np.std(x)))
        # data2=data2.apply(lambda x : (x-np.mean(x))/(np.std(x)))
        # data3=data3.apply(lambda x : (x-np.mean(x))/(np.std(x)))
        # data4=data4.apply(lambda x : (x-np.mean(x))/(np.std(x)))
        autoco_domain1[j,:] = tremor_utils.autocorr_domain(data1, N, fs)
        autoco_domain2[j,:] = tremor_utils.autocorr_domain(data2, N, fs)
        autoco_domain3[j,:] = tremor_utils.autocorr_domain(data3, N, fs)
        autoco_domain4[j,:] = tremor_utils.autocorr_domain(data4, N, fs)
        autocorr_peak_a[j,:]  = tremor_utils.auto_peak_xy(data4, N, fs, peak_num=5)

        i = i + windowsize // overlapping - 1
        j = j + 1
    
    fea_whole = np.c_[peaks_normal, fea_sampley, fea_samplex, fea_infory, fea_inforx, peaks_abnormal,fea_autoy,fea_auto_num,meandif,vardif]
    f1 = np.c_[time_axiscof,fft_peak_a,psd_peak_a,autocorr_peak_a,fre_df,PSDEnergy_XYZ,spectrumConcentration]
    # 20，25，26，24
    fx = np.c_[time_domain1,fre_domain1,psd_domain1, autoco_domain1] 
    fy = np.c_[time_domain2,fre_domain2,psd_domain2, autoco_domain2] 
    fz = np.c_[time_domain3,fre_domain3,psd_domain3, autoco_domain3] 
    fa = np.c_[time_domain4,fre_domain4,psd_domain4, autoco_domain4] 
    
    Feat = np.c_[fea_whole,f1, fx, fy, fz, fa]

    print(Feat.shape)
    print(Feat)
    Feat2 = np.zeros((j, Feat.shape[1]))  # 后一个参数为特征种类加一 28 38 16 45
    Feat2[0:j, :] = Feat[0:j, :]
    Feat2 = pd.DataFrame(Feat2)
    return Feat2