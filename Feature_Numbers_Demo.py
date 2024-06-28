#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: macpro
"""

#特征递增测试特征value
# shap 三分类 7,9,14[[0.6195652173913043, 0.6304347826086957, 0.5869565217391305, 0.5760869565217391, 0.5978260869565217, 0.5978260869565217], [0.6421052631578947, 0.7157894736842105, 0.6631578947368421, 0.6947368421052632, 0.6421052631578947, 0.6947368421052632], [0.5789473684210527, 0.5964912280701754, 0.5263157894736842, 0.5964912280701754, 0.631578947368421, 0.6140350877192983]]
# lgb 
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
path = "/Volumes/T7 Shield/TBME_code3.0/feature/"
pd_url= path + "pd_14activity_feature_label_right_level.csv"
hc_url = path + "hc_14activity_long_short_636feature_right_side.csv"
pdhc_url = path + "pdhc_14activity_long_short_636feature_right_side.csv"
save_path = "/Volumes/T7 Shield/TBME_code3.0/pd_fine_shap_importance/"
data_pd = pd.read_csv(pd_url, header=0).dropna()  # 读取总做数据文件
data_hc = pd.read_csv(hc_url, header=0).dropna()  # 读取总做数据文件
data_pdhc = pd.read_csv(pdhc_url, header=0).dropna()  # 读取总做数据文件
data_pd['Severity_Level'] = data_pd['Severity_Level'].map({0: 0, 1: 0, 2:1, 3:2, 4: 2})
data_hc['Severity_Level'] = data_hc['Severity_Level'].map({6: 4})
data_pdhc['Severity_Level'] = data_pdhc['Severity_Level'].map({7: 4})
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
 
#%
label =data.columns[-1]
subject_id=data.columns[-3]
activity_id=data.columns[-2]
features=data.columns[:-3].tolist()
chooselevel=[0,1,2]#选择等级
data_choose=data.loc[data[label].isin(chooselevel)]

# personlist=list(set(data_choose[subject_id].values.tolist())) 

plist = [9]
ala = []
for p in plist:
    # feature_data = pd.read_csv(url2+"early_0_1_shap_value_importance_sort_activity{}.csv".format(p), header=None)    
    feature_data = pd.read_csv(save_path+"shap_feature_activity{}.csv".format(p), header=None)    
    # feature_data = pd.read_csv(url1+"hc_pd_feature_importance_severity_level_activity{}.csv".format(p), header=None)
    # selected_feature = feature_data.iloc[:,0]
    selected_feature = feature_data.iloc[:, -1]
    selected_feature = np.array(selected_feature).tolist()
    # selected_feature = feature_new_sort
    print(selected_feature[:96])
    
    data_p = data_choose[data_choose["activity_label"] == p]

    # 标准化
    X_data = data_p[features]
    X_label = data_p[data_p.columns[-3:].tolist()]
    X_data = (X_data-X_data.mean())/(X_data.std()) 
    data_singleactivity = pd.concat([X_data,X_label],axis=1)
    data_singleactivity = data_singleactivity.reset_index(drop=True)
    
    plot_x=[] #特征个数列表
    plot_y=[] #特征个数对应的准确度评分
    for i in range(16,190,15):
    # for i in range(15,100,15):
        #data_singleactivity = data_singleactivity.iloc[:,:-i+3]
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
            datatrain = data_singleactivity.loc[data_singleactivity[subject_id].isin(trainperson)]
    
            X_train = datatrain[selected_feature[:i]]
            y_train = datatrain[label]
            #标准化
            X_train = (X_train-X_train.mean())/(X_train.std()) 
            
            X_test = datatestlf[selected_feature[:i]]
            y_test = datatestlf[label]
            #标准化
            X_test = (X_test-X_train.mean())/(X_train.std()) 
            
            # Creating the dataframe 
            train_data = lgb.Dataset(X_train,y_train)
            validation_data = lgb.Dataset(X_test,y_test)
            # Define parameters for the model,
            params = {
                    'learning_rate': 0.03,
                    'min_child_samples': 3,  # 子节点最小样本个数
                    'max_depth': 150,  # 树的最大深度
                    'lambda_l1': 0.3,  # 控制过拟合的超参数
                    'boosting': 'gbdt',
                    'objective': 'multiclass',
                    'num_boost_round': 50,  # 决策树的最大数量
                    'metric': 'multi_logloss',
                    'num_class': 3,
                    'feature_fraction': .9,  # 每次选取百分之75的特征进行训练，控制过拟合
                    'bagging_fraction': .85,  # 每次选取百分之85的数据进行训练，控制过拟合
                    'seed': 0,
                    'num_threads': 20,
                    'early_stopping_rounds': 30,  # 当验证集在训练一百次过后准确率还没提升的时候停止
                    'verbose': -1,
                    # 'num_leaves': 16, #2分类
                    'num_leaves': 64, #3分类
                    }
            
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
            
        pre = pd.DataFrame(pre)
        test = pd.DataFrame(test)
        
        cm = confusion_matrix(test, pre)   #计算混淆矩阵值
        print(cm)
        cm = pd.DataFrame(cm)
        # cm.to_csv('lgb_output.csv', mode='a', header=False, index=False)
        # pre_rec_fscore = classification_report(test, pre, output_dict=True)
        # print(pre_rec_fscore)
        report = classification_report(test,pre,output_dict=True)
        df = pd.DataFrame(report).transpose()
        df = df.round(4)
        print(df)
        # cm = pd.DataFrame(cm)
        # df.to_csv('lgb_output.csv', mode='a', header=False, index=False)

        test_array = np.array(test)
        pre_array = np.array(pre)
        score = (test_array == pre_array).mean()
        accuracy = accuracy_score(test, pre)
        accuracy = accuracy.round(4)
        print("feature number is {}, score is {}, accuracy is {}".format(i, score, accuracy))
        plot_x.append(len(X_train.columns))
        plot_y.append(score)
    print(plot_x)
    print(plot_y)
    ala.append(plot_y)
    plt.plot(plot_x, plot_y)
    # plt.xlim(15,0)
    plt.xticks(plot_x, plot_x)
    plt.title("Effect of the number of features for activity_{}".format(p),fontsize=15)
    plt.xlabel("Number of features",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.savefig(save_path + "lgb_sorted_numfeat{}.jpg".format(p),dpi=600)
    plt.show()
print(ala)        
