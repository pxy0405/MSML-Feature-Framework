#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:33:35 2024

@author: macpro
"""

#读取文件 PD_data_summary 名为 Patients information 的sheet， 第一行为header
import pandas as pd

# Define the path to the Excel file
file_path = '/Volumes/T7 Shield/0512-100pd+35hc/PD_data_summary.xlsx'  # Update this path if necessary

# Load the specified sheet
sheet_name = 'Patients information'
data = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

# Drop the first row (index 0)
data = data.drop(0)

# Reset the index
data = data.reset_index(drop=True)

# Display the first few rows of the data
print(data.head())


#删除第二行， 统计Severity_Level列 分别为 0+1， 2， 3+4 时候的 Gender, Age, Height, Weight, 均值与标准差，忽略-1的值

# Filter rows where Severity_Level is 0 or 1
filtered_data_mild = data[(data['Severity_Level'] == 0) | (data['Severity_Level'] == 1)]
filtered_data_moderate = data[(data['Severity_Level'] == 2)]
filtered_data_severe = data[(data['Severity_Level'] == 3) | (data['Severity_Level'] == 4)]

# Replace -1 with NaN to ignore these values in calculations
filtered_data_mild.replace(-1, pd.NA, inplace=True)
filtered_data_moderate.replace(-1, pd.NA, inplace=True)
filtered_data_severe.replace(-1, pd.NA, inplace=True)

# Calculate mean and standard deviation for the specified columns, ignoring NaN values
summary_mild = filtered_data_mild[['Gender', 'Age', 'Height', 'Weight']].agg(['mean', 'std'])
summary_moderate = filtered_data_moderate[['Gender', 'Age', 'Height', 'Weight']].agg(['mean', 'std'])
summary_severe = filtered_data_severe[['Gender', 'Age', 'Height', 'Weight']].agg(['mean', 'std'])

# Display the summary statistics
print("---------------------------------")
print(summary_mild)
print("---------------------------------")
print(summary_moderate)
print("---------------------------------")
print(summary_severe)


###old health 


###young health



###Gender统计数目


#%%导入特征文件，Samples统计
path = "/Volumes/T7 Shield/TBME_code3.0/feature/"
# hc feature file_path
hc_feature = pd.read_csv(path + "hc_14activity_long_short_636feature_right_side.csv",header=0)
# pdhc feature file_path
pdhc_feature = pd.read_csv(path + "pdhc_14activity_long_short_636feature_right_side.csv",header=0)
# pd feature file_path
pd_feature = pd.read_csv(path + "pd_14activity_feature_label_right_level.csv",header=0)

# Filter rows where Severity_Level is 0 or 1
pd_feature_mild = pd_feature[(pd_feature['Severity_Level'] == 0) | (pd_feature['Severity_Level'] == 1)]
pd_feature_moderate = pd_feature[(pd_feature['Severity_Level'] == 2)]
pd_feature_severe = pd_feature[(pd_feature['Severity_Level'] == 3) | (pd_feature['Severity_Level'] == 4)]
print(len(pd_feature_mild))
print(len(pd_feature_moderate))
print(len(pd_feature_severe))