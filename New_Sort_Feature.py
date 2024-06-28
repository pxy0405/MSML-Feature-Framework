#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:38:46 2024

@author: macpro
"""

acc_feature = []

gro_feature = []

selected_early_detect_feature = ['gyro_t_skew_a','gyro_a_amp_y','gyro_t_max_a','gyro_peaks_normal'] #0.8235lgb
# selected_early_detect_feature = ['gyro_t_skew_a','gyro_f_peakXY1','gyro_a_amp_y']

#%%left & right 舍弃，无论如何，左手数据的引入会给模型带来困难
#feature importance by lgb (patients_based)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# from autofeatselect import CorrelationCalculator, FeatureSelector, AutoFeatureSelect

#读取原始文件
url_r="/Volumes/T7 Shield/augment/feature_label_new.csv"
url_l="/Volumes/T7 Shield/augment/feature_label_left_level.csv"
url1="/Users/macpro/huawei_spyder/"
data = pd.read_csv(url_r, header=0).dropna()  # 读取总做数据文件
data['severity_level'] = data['severity_level'].map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3})
where_are_nan = np.isnan(data)
where_are_inf = np.isinf(data)
#nan替换成0,inf替换成nan
data[where_are_nan] = 0
data[where_are_inf] = 0
data.drop(data.columns[0], axis=1, inplace=True)#删除第一列
label =data.columns[-1]
subject_id=data.columns[-3]
activity_id=data.columns[-2]
features=data.columns[:-3].tolist()
chooselevel=[0,1,2,3]#选择等级
data=data.loc[data[label].isin(chooselevel)]
personlist=list(set(data[subject_id].values.tolist())) 

data_l = pd.read_csv(url_l, header=0).dropna()  # 读取总做数据文件
data_l['Severity_Level'] = data_l['Severity_Level'].map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3})
where_are_nan = np.isnan(data_l)
where_are_inf = np.isinf(data_l)
#nan替换成0,inf替换成nan
data_l[where_are_nan] = 0
data_l[where_are_inf] = 0
data_l.drop(data_l.columns[0], axis=1, inplace=True)#删除第一列
label_l =data_l.columns[-1]
subject_id_l=data_l.columns[-3]
activity_id_l=data_l.columns[-2]
features_l=data_l.columns[:-3].tolist()
# chooselevel=[0,1,2,3]#选择等级
data_l=data_l.loc[data_l[label_l].isin(chooselevel)]
personlist_l=list(set(data_l[subject_id_l].values.tolist())) 

X_data = data[features]
X_label = data[data.columns[-3:].tolist()]
# 标准化
X_data = (X_data-X_data.mean())/(X_data.std()) 
data = pd.concat([X_data,X_label],axis=1)

X_data_l = data_l[features_l]
X_label_l = data_l[data_l.columns[-3:].tolist()]
# 标准化
X_data_l = (X_data_l-X_data_l.mean())/(X_data_l.std()) 
data_l = pd.concat([X_data_l,X_label_l],axis=1)

for p in range(3,4):
    #选择活动划分训练&验证
    data_singleactivity = data[data["activity_label"] == p]
    #选择病人划分训练&验证
    personlist=list(set(data_singleactivity[subject_id].values.tolist()))
    falselist = []
    falsepredict = []
    result=[]
    true=[]
    test=[]
    pre=[]
    feature_df = pd.DataFrame()
    
    #选择活动划分训练&验证
    data_singleactivity_l = data_l[data_l["activity_label"] == p]
    #选择病人划分训练&验证
    personlist_l=list(set(data_singleactivity_l[subject_id_l].values.tolist()))
    falselist_l = []
    falsepredict_l = []
    result_l=[]
    true_l=[]
    test_l=[]
    pre_l=[]
    feature_df_l = pd.DataFrame()
    for pe in personlist:
        #划分训练&验证
        trainperson=personlist[:]
        testperson=pe
        trainperson.remove(pe)
        datatestlf = data_singleactivity.loc[data_singleactivity[subject_id] == testperson]
        # datatrain = data_train.loc[data_train[subject_id].isin(trainperson)]
        datatrain = data_singleactivity.loc[data_singleactivity[subject_id].isin(trainperson)]
        
        datatestlf_l = data_singleactivity_l.loc[data_singleactivity_l[subject_id_l] == testperson]
        # datatrain = data_train.loc[data_train[subject_id].isin(trainperson)]
        datatrain_l = data_singleactivity_l.loc[data_singleactivity_l[subject_id_l].isin(trainperson)]

        X_test = datatestlf[features]
        y_test = datatestlf[label]
        X_train = datatrain[features]
        y_train = datatrain[label]
    
        # Creating the dataframe 
        train_data = lgb.Dataset(X_train,y_train)
        validation_data = lgb.Dataset(X_test, y_test)
        
        X_test_l = datatestlf_l[features_l]
        y_test_l = datatestlf_l[label_l]
        X_train_l = datatrain_l[features_l]
        y_train_l = datatrain_l[label_l]
    
        # Creating the dataframe 
        train_data_l = lgb.Dataset(X_train_l,y_train_l)
        validation_data_l = lgb.Dataset(X_test_l, y_test_l)
        # Define parameters for the model
        params = {
                'learning_rate': 0.03,
                'min_child_samples': 3,  # 子节点最小样本个数
                'max_depth': 300,  # 树的最大深度
                'lambda_l1': 0.3,  # 控制过拟合的超参数
                'boosting': 'gbdt',
                'objective': 'multiclass',
                'n_estimators': 100,  # 决策树的最大数量
                'metric': 'multi_error',
                'num_class': 4,
                'feature_fraction': .9,  # 每次选取百分之75的特征进行训练，控制过拟合
                'bagging_fraction': .85,  # 每次选取百分之85的数据进行训练，控制过拟合
                'seed': 0,
                'num_threads': 20,
                'early_stopping_rounds': 15,  # 当验证集在训练一百次过后准确率还没提升的时候停止
                'verbose': -1,
                'num_leaves': 128,
                }
        # num_round = 300
        
        # Train the LightGBM model  
        model = lgb.train(params, train_data, valid_sets=[validation_data])
        # 对测试集进行预测
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=1) 
        pre_mode = stats.mode(predictions)[0]
        
        # predictions1 = [pre_mode for i in range(len(predictions))]
        # pre_label = pre_mode
        # true_label = test_data["label_hy"].iloc[1]
        pre.append(pre_mode)
        y_test = y_test.reset_index(drop=True)
        test.append(y_test[0])
        
        # Train the LightGBM model  
        model_l = lgb.train(params, train_data_l, valid_sets=[validation_data_l])
        # 对测试集进行预测
        predictions_l = model_l.predict(X_test_l)
        predictions_l = np.argmax(predictions_l, axis=1) 
        pre_mode_l = stats.mode(predictions_l)[0]
        
        # predictions1 = [pre_mode for i in range(len(predictions))]
        # pre_label = pre_mode
        # true_label = test_data["label_hy"].iloc[1]
        pre_l.append(pre_mode_l)
        y_test_l = y_test_l.reset_index(drop=True)
        test_l.append(y_test_l[0])
        if pre_mode==y_test[0]:
            None
        else:
            falselist.append(pe)
            falsepredict.append(y_test[0])
        pre_all =  [max(x, y) for x, y in zip(pre, pre_l)]
    print(test)
    print(pre_all)
    print(pre)
    print(pre_l)
    
    pre_all = pd.DataFrame(pre_all)
    test = pd.DataFrame(test)
    pre_l = pd.DataFrame(pre_l)
    pre = pd.DataFrame(pre)
    result_matrix = pd.concat(test, pre_all, pre_l, pre)
    
    cm = confusion_matrix(test, pre_all)   #计算混淆矩阵值
    pre_rec_fscore = classification_report(test, pre_all, output_dict=True)
   
    report = classification_report(test,pre_all,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.round(4)
    print(df)
    # cm = pd.DataFrame(cm)
    df.to_csv('patients_based_score_3classification.csv', mode='a', header=False, index=False)
    
    # Plot feature importance using Gain
    # lgb.plot_importance(model, importance_type="gain", figsize=(7,6),max_num_features=30, title="LightGBM Feature Importance (Gain)")
    # plt.show()
#%%NEW FEATURE SORT METHOD
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
feature_import ="/Volumes/T7 Shield/TBME_code3.0/pd_fine_shap_importance/"
feature_import1 = "/Volumes/T7 Shield/TBME_code3.0/pd_12_shap_importance/"
feature_import2 = "/Volumes/T7 Shield/TBME_code3.0/pd_1_34_shap_importance/"
feature_import3 = "/Volumes/T7 Shield/TBME_code3.0/pd_2_34_shap_importance/"
# url_l="/Volumes/T7 Shield/augment/feature_label_left_level.csv"
# url1="/Users/macpro/huawei_spyder/"
p=9
shap_feature = pd.read_csv(feature_import+"shap_feature_activity{}.csv".format(p), header=None)
shap_feature12 = pd.read_csv(feature_import1+"shap_feature_activity{}.csv".format(p), header=None)
shap_feature134 = pd.read_csv(feature_import2+"shap_feature_activity{}.csv".format(p), header=None)
shap_feature234 = pd.read_csv(feature_import3+"shap_feature_activity{}.csv".format(p), header=None)
lgb_feature = pd.read_csv(feature_import+"lgb_feature_activity{}.csv".format(p),header=None)

shap = pd.DataFrame()
shap12 = pd.DataFrame()
shap134 = pd.DataFrame()
shap234 = pd.DataFrame()
lgb = pd.DataFrame()
shap_sum = shap_feature.iloc[:, -4]
shap_std = shap_feature.iloc[:, -2]
shap_name = shap_feature.iloc[:, -1]
shap_name12 = shap_feature12.iloc[:, -1]
shap_name134 = shap_feature134.iloc[:, -1]
shap_name234 = shap_feature234.iloc[:, -1]
shap_sum12 = shap_feature12.iloc[:, -4]
shap_sum134 = shap_feature134.iloc[:, -4]
shap_sum234 = shap_feature234.iloc[:, -4]
shap["feature"] = shap_name
shap["shap_sum"] = shap_sum
shap["shap_std"] = shap_std
shap12["feature"] = shap_name12
shap134["feature"] = shap_name134
shap234["feature"] = shap_name234
shap12["shap_sum12"] = shap_sum12
shap["shap_index"] = shap.index
shap12["shap_index12"] = shap12.index
shap134["shap_sum134"] = shap_sum134
shap134["shap_index134"] = shap134.index
shap234["shap_sum234"] = shap_sum234
shap234["shap_index234"] = shap234.index
lgb_mean = lgb_feature.iloc[:, -2]
lgb_std = lgb_feature.iloc[:, -1]
lgb_name = lgb_feature.iloc[:, 0]
lgb["feature"] = lgb_name
lgb["lgb_mean"] = lgb_mean
lgb["lgb_std"] = lgb_std
print(shap12)
feature = pd.merge(shap, lgb, on='feature')
feature = pd.merge(feature, shap12, on='feature')
feature = pd.merge(feature, shap134, on='feature')
feature = pd.merge(feature, shap234, on='feature')
print(feature)
feature_value = feature[["shap_sum",'shap_std',"lgb_mean","lgb_std"]]
# feature_value = feature[["shap_sum","shap_sum12","shap_sum134"]]
feature_index = feature[["shap_sum"]]
# feature_index = feature[["shap_index","shap_index12","shap_index134","shap_index234"]]

feature["index_sum"] = feature_index.sum(axis=1)
feature["value_sum"] = feature_value.sum(axis=1)
print(feature_index)
# 创建一个StandardScaler对象
scaler = StandardScaler()
# 对DataFrame进行标准化（按列归一化）
feature_array = scaler.fit_transform(feature_value)
feature["sort_value"] = feature_array[:,0]+feature_array[:,3]+feature_array[:,2]
# feature["sort_value1"] = feature_index[:,0]+feature_index[:,1]+feature_index[:,2]
feature_new_sort = feature[["feature","index_sum","value_sum"]]
feature_new_sort = feature_new_sort.sort_values(by='index_sum', ascending=True)
feature_new_sort1 = feature_new_sort.sort_values(by='value_sum', ascending=True)
feature_new_sort9 = feature_new_sort
print(feature_new_sort)
print(feature_new_sort1)
print(feature_array)

#%%多分类器roc 曲线
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from autofeatselect import CorrelationCalculator, FeatureSelector, AutoFeatureSelect

#读取原始文件
url="/Volumes/T7 Shield/augment/feature_label_new.csv"
# url_l="/Volumes/T7 Shield/augment/feature_label_left_level.csv"
url1="/Users/macpro/huawei_spyder/"
data = pd.read_csv(url, header=0).dropna()  # 读取总做数据文件
data['severity_level'] = data['severity_level'].map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3})
where_are_nan = np.isnan(data)
where_are_inf = np.isinf(data)
#nan替换成0,inf替换成nan
data[where_are_nan] = 0
data[where_are_inf] = 0
data.drop(data.columns[0], axis=1, inplace=True)#删除第一列
label =data.columns[-1]
subject_id=data.columns[-3]
activity_id=data.columns[-2]
features=data.columns[:-3].tolist()
chooselevel=[0,1,2,3]#选择等级
data=data.loc[data[label].isin(chooselevel)]
personlist=list(set(data[subject_id].values.tolist())) 

# X_data = data[features]
# X_label = data[data.columns[-3:].tolist()]
# # 标准化
# X_data = (X_data-X_data.mean())/(X_data.std()) 
# data = pd.concat([X_data,X_label],axis=1)

for p in range(3,4):
    #选择活动划分训练&验证
    data_singleactivity = data[data["activity_label"] == p]
    
    X_data = data_singleactivity[features]
    X_label = data_singleactivity[data_singleactivity.columns[-3:].tolist()]
    X_data = (X_data-X_data.mean())/(X_data.std()) 
    data_singleactivity = pd.concat([X_data,X_label],axis=1)
    #选择病人划分训练&验证
    personlist=list(set(data_singleactivity[subject_id].values.tolist()))
    falselist = []
    falsepredict = []
    result=[]
    true=[]
    test=[]
    pre=[]
    feature_df = pd.DataFrame()
    for pe in personlist:
        #划分训练&验证
        trainperson=personlist[:]
        testperson=pe
        trainperson.remove(pe)
        datatestlf = data_singleactivity.loc[data_singleactivity[subject_id] == testperson]
        # datatrain = data_train.loc[data_train[subject_id].isin(trainperson)]
        datatrain = data_singleactivity.loc[data_singleactivity[subject_id].isin(trainperson)]

        X_test = datatestlf[features]
        y_test = datatestlf[label]
        X_train = datatrain[features]
        y_train = datatrain[label]
    
        # Creating the dataframe 
        train_data = lgb.Dataset(X_train,y_train)
        validation_data = lgb.Dataset(X_test, y_test)
        # Define parameters for the model
        params = {
                'learning_rate': 0.03,
                'min_child_samples': 3,  # 子节点最小样本个数
                'max_depth': 300,  # 树的最大深度
                'lambda_l1': 0.3,  # 控制过拟合的超参数
                'boosting': 'gbdt',
                'objective': 'multiclass',
                'n_estimators': 100,  # 决策树的最大数量
                'metric': 'multi_error',
                'num_class': 4,
                'feature_fraction': .9,  # 每次选取百分之75的特征进行训练，控制过拟合
                'bagging_fraction': .85,  # 每次选取百分之85的数据进行训练，控制过拟合
                'seed': 0,
                'num_threads': 20,
                'early_stopping_rounds': 15,  # 当验证集在训练一百次过后准确率还没提升的时候停止
                'verbose': -1,
                'num_leaves': 128,
                }
        
        
        # num_round = 300
        
        # Train the LightGBM model  
        # model = lgb.train(params, train_data, valid_sets=[validation_data])
        
        ####Train the xgboost model 
        model=xgb.XGBClassifier(n_estimators=5,max_depth=2)
        
        ##Train logistic model
        # model = LogisticRegression()
        
        ##Train knn model
        # model = KNeighborsClassifier(n_neighbors=7)
        
        ##Train adaboost model
        # model = AdaBoostClassifier(n_estimators=100, random_state=0)
        # #logreg.fit(X_train, y_train.values.reshape(-1))
        # X_train = X_train.values
        # X_test = X_test.values
        print("1")
        model.fit(X_train,y_train)
        # 对测试集进行预测
        predictions = model.predict(X_test)
        # predictions = np.argmax(predictions, axis=1) #当lgb,xgboost输入为独热编码的时候需要
        # predictions = np.argmax(predictions) 
        pre_mode = stats.mode(predictions)[0]
        
        
        pre.append(pre_mode)
        y_test = y_test.reset_index(drop=True)
        test.append(y_test[0])
        if pre_mode==y_test[0]:
            None
        else:
            falselist.append(pe)
            falsepredict.append(y_test[0])
        
    pre = pd.DataFrame(pre)
    test = pd.DataFrame(test)
    
    cm = confusion_matrix(test, pre)   #计算混淆矩阵值
    pre_rec_fscore = classification_report(test, pre, output_dict=True)
   
    # logit_roc_auc = roc_auc_score(test, pre)
    # fpr, tpr, thresholds = roc_curve(test, pre[:,1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()
    
    report = classification_report(test,pre,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.round(4)
    print(df)
    # cm = pd.DataFrame(cm)
    df.to_csv('patients_based_score_3classification.csv', mode='a', header=False, index=False)
    
    # Plot feature importance using Gain
    # lgb.plot_importance(model, importance_type="gain", figsize=(7,6),max_num_features=30, title="LightGBM Feature Importance (Gain)")
    # plt.show()

