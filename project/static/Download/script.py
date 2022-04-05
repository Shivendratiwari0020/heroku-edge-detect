#!/usr/bin/env python
# coding: utf-8

# In[51]:


# Inputs: Unseen data,  segment Models, Segments, rules, base model(name it base_model.sav for backend as this can be any model but wille be the best model on complete validation set.) )


# In[44]:


# Importing the packages
import re
import copy
import pickle
import json
import pandas as pd

# In[45]:


# Data Preprocessing
df = pd.read_csv("UnseenData.csv")
# Must run on all
df_cat_ = df.select_dtypes(exclude=['float64', 'int64'])
df_int_ = df.select_dtypes(include=['float64', 'int64'])
# Missing values
df_cat = df_cat_.fillna('NA')
df_int = df_int_.fillna(0)
# One hot encoding for categorical features.
df_cat = pd.get_dummies(df_cat)
df = pd.concat([df_cat, df_int], axis=1)

# In[46]:


# Reading the segment and rules
f = open('Segment_Rules.json')
seg_rules_dict = json.load(f)

# In[47]:


# Creating a dictionary for segment and model names
temp_seg_dict = seg_rules_dict["segment"]
seg_models = {}
for i in range(len(temp_seg_dict)):
    temp = temp_seg_dict[i]['segment' + str(i)]['InputFileds'][0]['modelevaluated']
    seg_models.update({'segment' + str(i): temp})

# In[48]:


# Loading the models
import joblib
import os

seg_models_dic = {}
for i in range(len(temp_seg_dict)):
    filename = seg_models['segment' + str(i)] + '.sav'
    loaded_model = joblib.load(open(filename, 'rb'))
    model_name = seg_models['segment' + str(i)]
    seg_models_dic.update({model_name: loaded_model})

# loading the base model
filename = 'base_model.sav'
base_model = joblib.load(open(filename, 'rb'))


# In[49]:


# Functions to filter Data, Segmentation,Rules:

def filter_data(new_dict, X_test):
    empty_df = pd.DataFrame()
    if new_dict['condition'] == 'AND':
        empty_df = copy.deepcopy(df)
        for key in new_dict.keys():
            if key != 'condition':
                if new_dict[key][1] == '=':
                    empty_df = empty_df[empty_df[key] == new_dict[key][0]]
                if new_dict[key][1] == '<=':
                    empty_df = empty_df[empty_df[key] <= new_dict[key][0]]
                if new_dict[key][1] == '>=':
                    empty_df = empty_df[empty_df[key] >= new_dict[key][0]]
                if new_dict[key][1] == '<':
                    empty_df = empty_df[empty_df[key] < new_dict[key][0]]
                if new_dict[key][1] == '>':
                    empty_df = empty_df[empty_df[key] > new_dict[key][0]]

    if new_dict['condition'] == 'OR':
        for key in new_dict.keys():
            if key != 'condition':
                if new_dict[key][1] == '=':
                    empty_df = pd.concat([empty_df, X_test[X_test[key] == new_dict[key][0]]], axis=0)
                if new_dict[key][1] == '<=':
                    empty_df = pd.concat([empty_df, X_test[X_test[key] <= new_dict[key][0]]], axis=0)
                if new_dict[key][1] == '>=':
                    empty_df = pd.concat([empty_df, X_test[X_test[key] >= new_dict[key][0]]], axis=0)
                if new_dict[key][1] == '<':
                    empty_df = pd.concat([empty_df, X_test[X_test[key] < new_dict[key][0]]], axis=0)
                if new_dict[key][1] == '>':
                    empty_df = pd.concat([empty_df, X_test[X_test[key] > new_dict[key][0]]], axis=0)

    return empty_df


def SegmentModelling(df, seg_models_dic, temp_seg_dict, seg_models):
    modeled_df = pd.DataFrame()
    for i in range(len(temp_seg_dict)):
        dict_x = {}
        x = temp_seg_dict[i]['segment' + str(i)]['InputFileds'][0]
        del x['modelevaluated']
        dict_x.update({1: x})
        temp_dict = dict()

        for value in dict_x.values():
            if value['selectVariable'] in df_int_.columns:
                temp_dict[value['selectVariable']] = [float(value['evaluationValue'])]
                temp_dict[value['selectVariable']].append(value['sampling'])
                temp_dict[value['selectVariable']].append(value['binaryOperation'])
                temp_dict['condition'] = value['binaryOperation']
            if value['selectVariable'] in df_cat_.columns:
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])] = [1]
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(value['sampling'])
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(
                    value['binaryOperation'])
                temp_dict['condition'] = value['binaryOperation']

        X_filter = filter_data(temp_dict, df)
        rows = df.shape[0]
        if X_filter.shape[0] > 0 & X_filter.shape[0] < rows:
            df = pd.concat([df, X_filter]).drop_duplicates(keep=False)
            model_name = seg_models['segment' + str(i)]
            model = seg_models_dic[model_name]
            y_filter = model[0].predict(X_filter)
            X_filter['target_variable'] = y_filter
            modeled_df = modeled_df.append(X_filter)
    if rows > 0:
        y_df = base_model[0].predict(df)
        df['target_variable'] = y_df
        modeled_df = modeled_df.append(df)

        return modeled_df


def ApplyingRules(temp_rules_dict, modeled_df):
    for i in range(len(temp_rules_dict)):
        dict_x = {}
        temp = temp_rules_dict['Rules'][i]
        dict_x = {1: temp}
        temp_dict = dict()
        for value in dict_x.values():
            if value['selectVariable'] in df_int_.columns:
                temp_dict[value['selectVariable']] = [float(value['evaluationValue'])]
                temp_dict[value['selectVariable']].append(value['selectOperator'])
                temp_dict['condition'] = value['selectOperation']
            if value['selectVariable'] in df_cat_.columns:
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])] = [1]
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(value['selectOperator'])
                temp_dict['condition'] = value['selectOperation']
        X_filter = filter_data(temp_dict, modeled_df)
        modeled_df = pd.concat([modeled_df, X_filter]).drop_duplicates(keep=False)
        X_filter['target_variable'] = temp_rules_dict['Rules'][i]['selectedVariable']
        modeled_df = modeled_df.append(X_filter)

    return modeled_df


# In[50]:


modeled_df = SegmentModelling(df, seg_models_dic, temp_seg_dict, seg_models)

# In[51]:


temp_rules_dict = seg_rules_dict['rules']

# In[52]:


final_df = ApplyingRules(temp_rules_dict, modeled_df)

# In[53]:


final_df.to_csv("Predicted_UnseenData.csv")

# In[ ]:




