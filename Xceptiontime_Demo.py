#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#test xceptiontime
"""
Created on Sun May 19 15:42:59 2024

@author: macpro
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import shap
from tsai.all import *
# from autofeatselect import CorrelationCalculator, FeatureSelector, AutoFeatureSelect
shap.initjs()

#读取原始文件
path = "/Volumes/T7 Shield/TBME_code3.0/feature/"
pd_url= path + "pd_14activity_feature_label_right_level.csv"
hc_url = path + "hc_14activity_long_short_636feature_right_side.csv"
pdhc_url = path + "pdhc_14activity_long_short_636feature_right_side.csv"
save_path = "/Volumes/T7 Shield/TBME_code3.0/pd_hc_bi_shap_importance/"
data_pd = pd.read_csv(pd_url, header=0).dropna()  # 读取总做数据文件
data_hc = pd.read_csv(hc_url, header=0).dropna()  # 读取总做数据文件
data_pdhc = pd.read_csv(pdhc_url, header=0).dropna()  # 读取总做数据文件
data_pd['Severity_Level'] = data_pd['Severity_Level'].map({0: 0, 1: 0, 2: 1, 3:2, 4: 2})
data_hc['Severity_Level'] = data_hc['Severity_Level'].map({6: 4})
data_pdhc['Severity_Level'] = data_pdhc['Severity_Level'].map({7: 3})
data_hc.drop(data_hc.columns[0], axis=1, inplace=True)#删除第一列
data_pdhc.drop(data_pdhc.columns[0], axis=1, inplace=True)#删除第一列
data = pd.concat([data_pd, data_hc, data_pdhc], axis=0)
#%
where_are_nan = np.isnan(data)
where_are_inf = np.isinf(data)
#nan替换成0,inf替换成nan
data[where_are_nan] = 0
data[where_are_inf] = 0
data.drop(data.columns[0], axis=1, inplace=True)#删除第一列
# data.to_csv(path + "pd_hc_14activity.csv")

#%
label =data.columns[-1]
subject_id=data.columns[-3]
activity_id=data.columns[-2]
features=data.columns[:-3].tolist()
chooselevel=[0,1,2,3]#选择等级
data_choose=data.loc[data[label].isin(chooselevel)]
personlist=list(set(data_choose[subject_id].values.tolist())) 


# data = data[data["activity_label"] == 2]
# print(data)

#%
alist = [3,7,8,9,10,12,14]
aclist = [9]
accuracy_append = []
# plist = [3]
for p in aclist:
    #选择活动划分训练&验证
    print("Activity:",p)
    activity_p = p
    data_p = data_choose[data_choose["activity_label"] == p]
    
    #%
    X_data = data_p[features]
    X_label = data_p[data_p.columns[-3:].tolist()]
    # 标准化
    X_data = (X_data-X_data.mean())/(X_data.std()) 
    data = pd.concat([X_data,X_label],axis=1)
    data = data.reset_index(drop=True)
    #%
    # data_singleactivity = data
    t=0
    #%
    for i in [sample_feature, seg_feature, base_time_feature, time_frequency_feature, time_autocorr_feature, time_spec_feature]:
    # for i in [sample_feature3, seg_feature]:
        # for i in [fea_column]:
        #选择病人划分训练&验证
        t=t+1
        personlist=list(set(data[subject_id].values.tolist()))
        falselist = []
        falsepredict = []
        result=[]
        true=[]
        test=[]
        pre=[]
        feature_df = pd.DataFrame()
        m = 0
        shap_list=[]
        x_list = []
        shap_mode = []
        reshaped_list = []
        for pe in personlist:
            m = m+1
            # print(m)
            #划分训练&验证
            trainperson=personlist[:]
            testperson=pe
            trainperson.remove(pe)
            datatestlf = data.loc[data[subject_id] == testperson]
            # datatrain = data_train.loc[data_train[subject_id].isin(trainperson)]
            datatrain = data.loc[data[subject_id].isin(trainperson)]
    
            X_test = datatestlf[i]
            y_test = datatestlf[label]
            X_train = datatrain[i]
            y_train = datatrain[label]
        
            # 创建 TimeseriesGenerator
            print("--------------------------------")
            generator1 = TimeseriesGenerator(X_train, y_train, length=20, batch_size=32, stride=20)
            generator2 = TimeseriesGenerator(X_test, y_test, length=20, batch_size=32, stride=20)
            # 检查生成的数据形状
            X_train, y_train = generator1[0]
            X_test, y_test = generator2[0]
            print("--------------------------------")
            # X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
            
            #xceptiontime model
            dls = get_ts_dls(X_train, y_train, X_test, y_test)
            # 创建 XceptionTime 模型
            model = XceptionTime(dls.vars, dls.c)
            print("***************************")
            # 创建 Learner 对象
            learn = Learner(dls, model, metrics=accuracy)
            # 训练模型
            learn.fit_one_cycle(25, 1e-3)
            # 评估模型
            learn.plot_metrics()
            learn.validate()
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            
            predictions, targets = learn.get_preds()
            print(f'Accuracy: {accuracy(preds, targets).item():.4f}')
            print("################################################")
            print(pe)
            # predictions = np.argmax(predictions, axis=1) 
            # predictions = np.argmax(predictions,axis=1) 
            pre_mode = stats.mode(predictions)[0]
            pre.append(pre_mode)
            y_test = y_test.reset_index(drop=True)
            test.append(y_test[0])
            if pre_mode==y_test[0]:
                None
            else:
                falselist.append(pe)
                falsepredict.append(y_test[0])
            
        print(falselist)
        print(falsepredict)
        
        # x_df = pd.DataFrame(x_list)
        # x_df.columns = fea_column
        # b_list = []
        
        ###################################模型评测##############################
        pre = pd.DataFrame(pre)
        test = pd.DataFrame(test)
        cm = confusion_matrix(test, pre)   #计算混淆矩阵值
        pre_rec_fscore = classification_report(test, pre, output_dict=True)
        report = classification_report(test,pre,output_dict=True)
        accuracy = accuracy_score(test, pre)
        print(report)
        df = pd.DataFrame(report).transpose()
        df = df.round(4)
        accuracy = accuracy.round(4)
        print(df)
        # cm = pd.DataFrame(cm)
        # df.to_csv(path + "1_2_34/" + 'result_matrix{}.csv'.format(t), mode='a')

        #如何保存report的accuracy唯一一个值
        accuracy_append.append(accuracy)
print(accuracy_append)

