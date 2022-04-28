import ast
import copy
import io
import operator
from pyexpat import model
from time import sleep

import joblib
from django.conf.urls import static
from django.shortcuts import render
import datetime
import json
import os
import os.path
import re
import shutil
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import pickle

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# views test funtions
from django.core import serializers
# from rest_framework import serializers
from rest_framework.decorators import api_view

from app import models
import pandas as pd
import numpy as np
# Importing visualization packages
# import matplotlib.pyplot as plt
# import seaborn as sns
# Importing model building packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE


@api_view(['GET', 'POST', ])
def test_views(request):
    if request.method == 'GET':
        return HttpResponse("Get method")
    elif request.method == 'POST':
        return HttpResponse("Post method")


# Create your views here.
def home_view(request):
    return render(request, "index.html")


@api_view(['GET', 'POST', ])
def upload_file(request):
    try:
        print(request)
        if request.method == 'POST':
            uploaded_file = request.FILES["file"]
            print(uploaded_file)
            _dir = 'app/static/'
            fs = FileSystemStorage(_dir)
            fs.save(uploaded_file.name, uploaded_file)
            s = os.path.join(_dir, uploaded_file.name)
            p = os.path.splitext(s)[0]
            d = os.path.split(p)
            f = list(d)
            file_name = f[1]
            # parent_dir='app/'
            # path = os.path.join(parent_dir, file_name)
            # print(path)
            # os.makedirs(path)
            global Project_Folder
            Project_Folder = re.sub('.csv', '', file_name)
            if not os.path.exists(Project_Folder):
                os.makedirs(Project_Folder)

            if not os.path.exists(Project_Folder + r'/Models/'):
                os.makedirs(Project_Folder + r'/Models/')

            if not os.path.exists(Project_Folder + r'/Models/Supervised'):
                os.makedirs(Project_Folder + r'/Models/Supervised')

            if not os.path.exists(Project_Folder + r'/Models/Unsupervised'):
                os.makedirs(Project_Folder + r'/Models/Unsupervised')

            if not os.path.exists(Project_Folder + r'/Models/Time_Series'):
                os.makedirs(Project_Folder + r'/Models/Time_Series')

            if not os.path.exists(Project_Folder + r'/Results/'):
                os.makedirs(Project_Folder + r'/Results/')

            if not os.path.exists(Project_Folder + r'/Data/'):
                os.makedirs(Project_Folder + r'/Data/')
                v = os.getcwd()
                r = os.path.join(v, file_name)
                r = os.path.join(r, "Data")
                shutil.move(s, r)
                l = os.path.join(r, uploaded_file.name)
                print(l)
                reading_csv = pd.read_csv(l)
                print(reading_csv)
                coloumn = reading_csv.columns
                # lists = coloumn.tolist()
                # json_str = json.dumps(lists)
                # print(type(json_str))
                # print(json_str)
            jobs_res = models.Upload.objects.all().delete()
            jobs_re = models.Sai.objects.all().delete()
            # jobs_r = models.Image.objects.all().delete()
            jobs = models.Segment.objects.all().delete()
            job = models.Rules.objects.all().delete()


            current_date = datetime.datetime.now()
            Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
            upload_create = models.Upload(
                upload_id=Id_val,
                project_name=file_name,
                upload_file_path=l
            )
            upload_create.save()
            print("after")
        res = {"status": "success", "message": "upload created successfully"}
        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


@api_view(['GET', 'POST', ])
def upload_file_get(request):
    try:
        operations_res = json.loads(serializers.serialize("json", models.Upload.objects.all()))
        res = []
        print(operations_res)
        print(type(operations_res))
        a = operations_res[0]
        print(a)
        print(type(a))
        p = a["fields"]
        path = p["upload_file_path"]
        print(path)
        reading_csv = pd.read_csv(path)
        print(type(reading_csv), "==============")
        print(reading_csv)
        description_df = reading_csv.describe()
        print("+++++++++++++++++++++++++++++++++++++++++++")
        # description_df = df.describe(include='all')
        # excluded_columns = [i for i in df.columns if i not in description_df.columns]
        # eda_df = pd.concat([description_df.reset_index(),
        #                     pd.DataFrame(np.zeros((description_df.shape[0], len(excluded_columns))),
        #                                  columns=excluded_columns)], axis=1)
        # # return eda_df.rename(columns={'index':'Stat Name'})
        #
        # # description_df = self.df.describe(include='all')
        df_cat_ = reading_csv.select_dtypes(exclude=['float64', 'int64'])
        df_int_ = reading_csv.select_dtypes(include=['float64', 'int64'])
        # excluded_columns = [i for i in self.df.columns if i not in description_df.columns]
        # eda_df = pd.concat([description_df.reset_index(),pd.DataFrame(np.zeros((description_df.shape[0],len(excluded_columns))),columns = excluded_columns)],axis=1)
        j = df_int_.describe(percentiles =[0.95,.99],include='all')
        j = j.round(decimals=2)
        total = reading_csv.isnull().sum().sort_values(ascending=False)
        percent = (reading_csv.isnull().sum() / reading_csv.isnull().count() * 100).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
        p=missing_data[missing_data['Percent'] > 0.0]
        print("ssssssssssssssssssssssss",type(p))
        print(p)
        p=p.reset_index()
        p=p.rename(columns={'index':'Stat Name'})
        p=p.to_json(orient='records')
        pp=ast.literal_eval(p)
        print("sssssssssssssssssssssssssssssss")

        k = df_cat_.describe(include='all')
        j = j.reset_index()
        j = j.rename(columns={'index': 'Stat Name'})
        k = k.reset_index()
        k = k.rename(columns={'index': 'Stat Name'})
        print(j)
        jj = j.to_json(orient='records')
        jjj = ast.literal_eval(jj)
        kk = k.to_json(orient='records')
        kkk = ast.literal_eval(kk)
        print(type(jjj))
        print(type(kkk))
        print("++++++++++")
        print(k)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        excluded_columns = [i for i in reading_csv.columns if i not in description_df.columns]
        eda_df = pd.concat([description_df.reset_index(),
                            pd.DataFrame(np.zeros((description_df.shape[0], len(excluded_columns))),
                                         columns=excluded_columns)], axis=1)
        data = eda_df.rename(columns={'index': 'Stat Name'})
        print("============")
        print(data)
        print(type(data))
        print("++++++++++++")
        # eda_df = self.return_eda_table()
        data2 = data.to_json(orient='records')
        print("Testing", "/n", data2, "/n", "Testing ends")
        items = {}
        rem = ast.literal_eval(data2)
        items["body"] = data2
        b = items["body"]
        vall = {"body": items["body"]}
        re = []
        re.append(vall)
        # js = reading_csv.to_json(orient='columns')
        # print("=======",type(js))
        coloumn = reading_csv.columns
        lists = coloumn.tolist()
        json_str = json.dumps(lists)
        ress = json_str.strip('][').split(', ')
        item = {}
        for i in ress:
            # Id=i
            item["userId"] = i
            # item["body"]="sai prakash reddy"
            Val = {"userId": item["userId"]}
            res.append(Val)
            print(res)

            # res.append(Id)
        # res.append(ress)
        # jsonString = json.dumps(res)
        # print(type(jsonString))
        # print(jsonString)
        # res.append(jsonString)
        # for i in operations_res:
        #     print("--------", i)
        #     fields = i["fields"]
        #     fields['model_id'] = i["pk"]
        #     res.append(fields)
        res = {
            "status": "successs",
            "message": "details get is successsss",
            "data": res, "summary": rem, "NumericalDataSummary": jjj, "CategoricalDataSummary": kkk,"missingData":pp
        }
        return JsonResponse(res)
    except Exception as e:
        res = {
            "status": "failed",
            "message": str(e)
        }
        return JsonResponse(res)


def undersample(df, target_column, percentage_undersample):
    df_bad_loan0 = df[df[target_column] == 0]
    df_bad_loan1 = df[df[target_column] == 1]
    count_bad_loan0 = df_bad_loan0.shape[0]
    count_bad_loan1 = df_bad_loan1.shape[0]
    df_bad_loan0_under = df_bad_loan0.sample(int(round(count_bad_loan1 * (float(100 / percentage_undersample)), 0)),
                                             replace=True)
    df_ = pd.concat([df_bad_loan1, df_bad_loan0_under], axis=0).reset_index(drop=True)
    return df_


def upsample(df, target_column, percentage_oversample):
    df_bad_loan0 = df[df[target_column] == 0]
    df_bad_loan1 = df[df[target_column] == 1]
    count_bad_loan0 = df_bad_loan0.shape[0]
    count_bad_loan1 = df_bad_loan1.shape[0]
    df_bad_loan1_over = df_bad_loan1.sample(int(round(count_bad_loan0 * (float(percentage_oversample / 100)), 0)),
                                            replace=True)
    df_ = pd.concat([df_bad_loan0, df_bad_loan1_over], axis=0).reset_index(drop=True)
    return df_


# def My_function3(df, target_column):
#     y = df[target_column]
#     print(y, "yyyyyyyyyyyy")
#     X = df.drop(target_column, axis=1)
#     print(X, "xxxxxxxx")
#     X = pd.get_dummies(X)
#     print(X, "Xy xy")
#     # split_percentage = request_json["splitPercentage"]
#     # split_percentage = int(split_percentage) / 100
#     print(type(split_percentage))
#     print(split_percentage, "split percentage")
#     global X_train
#     global X_test
#     global y_train
#     global y_test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=None)
#     global optimize_hyperparameter
#     optimize_hyperparameter = False
#     print(X_test, X_train, y_train, y_test)


def My_function1(target_column, split_percentage):
    from app import models
    operations_res = json.loads(serializers.serialize("json", models.Upload.objects.all()))
    res = []
    a = operations_res[0]
    p = a["fields"]
    print(p)
    project_name = p["project_name"]
    print(project_name)
    path = p["upload_file_path"]
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(path)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    df = pd.read_csv(path)
    # global df_cat
    # global df_int
    df_cat = df.select_dtypes(exclude=['float64', 'int64'])
    df_int = df.select_dtypes(include=['float64', 'int64'])
    df_cat = df_cat.fillna('NA')
    df_int = df_int.fillna(0)
    df = pd.concat([df_cat, df_int], axis=1)
    print("======================================")
    y = df[target_column]
    print(y, "yyyyyyyyyyyy")
    X = df.drop(target_column, axis=1)
    print(X, "xxxxxxxx")
    X = pd.get_dummies(X)
    print(X, "Xy xy")
    # split_percentage = request_json["splitPercentage"]
    split_percentage = int(split_percentage) / 100
    print(type(split_percentage))
    print(split_percentage, "split percentage")
    # global X_train
    # global X_test
    # global y_train
    # global y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=None)
    global optimize_hyperparameter
    optimize_hyperparameter = False
    print(X_test, X_train, y_train, y_test)

    return df, project_name, df_int, df_cat, X_train, X_test, y_train, y_test


def My_function(ds):
    modelss = dict()
    if 'Logistic Regression' in ds:
        print("log")
        modelss['logistic_regression'] = logistic_regression()

    if 'Decision Tree' in ds:
        print("dec")
        modelss['decision_tree'] = decision_tree()
    if 'XGBoost' in ds:
        print("Xg")
        modelss['xgboost_classifier'] = xgboost_classifier()
    if 'Gradient Boosting Machines' in ds:
        print("Gbm")
        modelss['gbm_classifier'] = gbm_classifier()
    if 'Light GBM' in ds:
        print("light")
        modelss['lgbm_classifier'] = lgbm_classifier()
    if 'CATBOOST' in ds:
        print("Catboost")
        modelss['cb_classifier'] = cb_classifier()
    if 'Random Forest' in ds:
        print("rf")
        modelss['rf_classifier'] = rf_classifier()
    # print(modelss)

    return modelss


@api_view(['GET', 'POST', ])
def start_modeling(request):
    print(request)
    try:
        request_json = json.loads(request.body)
        print("++++++++++++++++++", request_json)
        print("======================================================================")
        print(type(request_json))
        dd = request_json["targetVariable"]
        print(dd)
        # dd = request_json["targetVariable"]
        ds = request_json['modelData1']
        print("sssssssssssssssssssssssssssss")
        print(ds)
        print(type(ds))
        # target_column = dd[1:-1]
        # split_percentage = request_json["splitPercentage"]
        # ff=My_function1(target_column,split_percentage)
        # ff=list(ff)
        # df=ff[0]
        # project_name=ff[1]
        # print("saisasasasasasasasasasasasa")
        # print(ff)
        # print(type(ff))
        # print(df,"+++++++++++++++++++++++++++++++++++++++++++++++")
        # print(project_name)
        print("sasasasasasasasa")
        from app import models
        operations_res = json.loads(serializers.serialize("json", models.Upload.objects.all()))
        res = []
        a = operations_res[0]
        p = a["fields"]
        print(p)
        project_name = p["project_name"]
        print(project_name)
        path = p["upload_file_path"]
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(path)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        df = pd.read_csv(path)
        df_cat = df.select_dtypes(exclude=['float64', 'int64'])
        df_int = df.select_dtypes(include=['float64', 'int64'])
        df_cat = df_cat.fillna('NA')
        df_int = df_int.fillna(0)
        df = pd.concat([df_cat, df_int], axis=1)
        print("====================================")
        print("======================")
        # c = df_cat
        # i = df_int
        # d = df
        # cc_df_cat = str(c)
        # ii_df_int = str(i)
        # dd_df = str(d)

        print("==================")
        print("=========================================")
        target_column = dd[1:-1]
        if df[target_column].dtypes == 'int64':
            df[target_column] = df[target_column]
            print("working till")
        else:
            df[target_column] = pd.Series(np.where(df[target_column].values ==
                                                   pd.DataFrame(df[target_column].value_counts()).sort_values(
                                                       by=target_column).index[1], 1, 0),
                                          df.index)
            print("in else condition")

        if request_json["sampling"] == "Oversample":
            percentage_oversample = 100
            print("oversample=========")
            df = upsample(df, target_column, percentage_oversample)
            print(df, "over sample")

        elif request_json["sampling"] == "Undersample":
            percentage_undersample = 100
            print("undersample=============")
            df = undersample(df, target_column, percentage_undersample)
            print(df, "under sample")
        else:
            df = df
        print("subham subham")
        y = df[target_column]
        print(y, "yyyyyyyyyyyy")
        X = df.drop(target_column, axis=1)
        print(X, "xxxxxxxx")
        X = pd.get_dummies(X)
        print(X, "Xy xy")
        split_percentag = request_json["splitPercentage"]
        split_percentage = int(split_percentag) / 100
        print(type(split_percentage))
        print(split_percentage, "split percentage")
        global X_train
        global X_test
        global y_train
        global y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=None)
        global optimize_hyperparameter
        optimize_hyperparameter = False
        print("==================================================================================")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(X_train)

        print("xtrainxtrain",type(X_test),type(y_test))
        path = "static"
        import shutil
        import os
        filepath = os.path.join(path, "edge")
#        shutil.rmtree(filepath)
        print("000000000000")
        print(filepath)
        print("0000000000000")
        os.makedirs(filepath, exist_ok=True)
        file = "X_test.csv"
        file1="y_test.csv"
        file_path = os.path.join(filepath, file)
        file_path1=os.path.join(filepath,file1)
        print(file_path)
        print(file_path1)
        X_test.to_csv(file_path)
        y_test.to_csv(file_path1)

        print("=====")
        print(y_test)

        # X_te = X_test
        # X_tr = X_train
        # y_te = y_test
        # y_tr = y_train
        # X_X_test = str(X_te)
        # X_X_train = str(X_tr)
        # y_y_test = str(y_te)
        # y_y_train = str(y_tr)
        # print(type(X_X_train))
        print("===============================================================================================")
        # models = dict()

        # def logistic_regression(params={"C": [0.01, 0.1, 1, 10, 100, 1000]},
        #                         folds=KFold(n_splits=5, shuffle=True, random_state=4)
        #                         ):
        #     scaler = StandardScaler()
        #
        #     train_columns = X_train.columns
        #     X_train_ = pd.DataFrame(data=scaler.fit_transform(X_train), columns=train_columns)
        #     X_test_ = pd.DataFrame(data=scaler.fit_transform(X_test), columns=train_columns)
        #     X_train_ = X_train_.fillna(0)
        #     X_test_ = X_test_.fillna(0)
        #
        #     if optimize_hyperparameter == False:
        #         params = {i: [j] for i, j in LogisticRegression().get_params().items()}
        #
        #     model_cv = GridSearchCV(estimator=LogisticRegression(),
        #                             param_grid=params,
        #                             scoring='roc_auc',
        #                             cv=folds,
        #                             n_jobs=-1,
        #                             verbose=1,
        #                             return_train_score=True)
        #     # perform hyperparameter tuning
        #     model_cv.fit(X_train_, y_train)
        #     print('Best ROC AUC score: ', model_cv.best_score_)
        #     # print the optimum value of hyperparameters
        #     print('Best hyperparameters: ', model_cv.best_params_)
        #
        #     cv_results = pd.DataFrame(model_cv.cv_results_)
        #     return (model_cv, cv_results)
        #
        # def decision_tree(param_grid={'max_depth': range(5, 15, 5), 'min_samples_leaf': range(50, 150, 50),
        #                               'min_samples_split': range(50, 150, 50), }):
        #     # Instantiate the grid search model
        #     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
        #
        #     dtree = DecisionTreeClassifier()
        #
        #     if optimize_hyperparameter == False:
        #         param_grid = {i: [j] for i, j in dtree.get_params().items()}
        #
        #     grid_search = GridSearchCV(estimator=dtree,
        #                                param_grid=param_grid,
        #                                scoring='roc_auc',
        #                                cv=3,
        #                                n_jobs=-1,
        #                                verbose=1)
        #
        #     # Fit the grid search to the data
        #     grid_search.fit(X_train, y_train)
        #
        #     print("Best roc auc score : ", grid_search.best_score_)
        #     print(grid_search.best_estimator_)
        #
        #     cv_results = pd.DataFrame(grid_search.cv_results_)
        #     return (grid_search, cv_results)
        #
        # def xgboost_classifier(folds=3,
        #                        param_grid={'learning_rate': [0.2, 0.6],
        #                                    'subsample': [0.3, 0.6, 0.9]}):
        #     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
        #
        #     xgb_model = XGBClassifier(max_depth=2, n_estimators=200)
        #
        #     if optimize_hyperparameter == False:
        #         param_grid = {i: [j] for i, j in xgb_model.get_params().items()}
        #
        #     model_cv = GridSearchCV(estimator=xgb_model,
        #                             param_grid=param_grid,
        #                             scoring='roc_auc',
        #                             cv=folds,
        #                             verbose=1,
        #                             return_train_score=True)
        #
        #     model_cv.fit(X_train, y_train)
        #     print("Best roc auc score : ", model_cv.best_score_)
        #     print(model_cv.best_estimator_)
        #
        #     # cv results
        #     cv_results = pd.DataFrame(model_cv.cv_results_)
        #     return (model_cv, cv_results)
        #
        # def gbm_classifier(parameters={
        #     "loss": ["deviance"],
        #     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        #     "min_samples_split": np.linspace(0.1, 0.5, 12),
        #     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        #     "max_depth": [3, 5, 8],
        #     "max_features": ["log2", "sqrt"],
        #     "criterion": ["friedman_mse", "mae"],
        #     "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        #     "n_estimators": [10]
        #     }, folds=10):
        #     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
        #
        #     gbm_model = GradientBoostingClassifier()
        #     print("gbmgbmgmbmmbgm")
        #     if optimize_hyperparameter == False:
        #         parameters = {i: [j] for i, j in gbm_model.get_params().items()}
        #     print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
        #     model_cv = GridSearchCV(gbm_model, parameters, n_jobs=-1, scoring='roc_auc',
        #                             cv=folds,
        #                             verbose=1,
        #                             return_train_score=True)
        #
        #     model_cv.fit(X_train, y_train)
        #     print("Best roc auc score : ", model_cv.best_score_)
        #     print(model_cv.best_estimator_)
        #
        #     # cv results
        #     cv_results = pd.DataFrame(model_cv.cv_results_)
        #     print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        #     print(cv_results)
        #     print("/////////////////////////")
        #     return (model_cv, cv_results)
        #
        # def lgbm_classifier(parameters={'learning_rate': [0.01], 'n_estimators': [8, 24],
        #                                 'num_leaves': [6, 8, 12, 16], 'boosting_type': ['gbdt'],
        #                                 'objective': ['binary'], 'seed': [500],
        #                                 'colsample_bytree': [0.65, 0.75, 0.8],
        #                                 'subsample': [0.7, 0.75], 'reg_alpha': [1, 2, 6],
        #                                 'reg_lambda': [1, 2, 6]}, folds=10):
        #     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
        #
        #     lgb_model = lgb.LGBMClassifier()
        #
        #     if optimize_hyperparameter == False:
        #         parameters = {i: [j] for i, j in lgb_model.get_params().items()}
        #
        #     model_cv = GridSearchCV(lgb_model, parameters, n_jobs=-1, scoring='roc_auc',
        #                             cv=folds,
        #                             verbose=1,
        #                             return_train_score=True)
        #
        #     model_cv.fit(X_train, y_train)
        #     print("Best roc auc score : ", model_cv.best_score_)
        #     print(model_cv.best_estimator_)
        #
        #     # cv results
        #     cv_results = pd.DataFrame(model_cv.cv_results_)
        #     return (model_cv, cv_results)
        #
        # def cb_classifier(parameters={'iterations': [500],
        #                               'depth': [4, 5, 6],
        #                               'loss_function': ['Logloss', 'CrossEntropy'],
        #                               'l2_leaf_reg': np.logspace(-20, -19, 3),
        #                               'leaf_estimation_iterations': [10],
        #                               'logging_level': ['Silent'],
        #                               'random_seed': [42]
        #                               }, folds=10):
        #
        #     #     (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
        #
        #     cb_model = CatBoostClassifier()
        #
        #     if optimize_hyperparameter == False:
        #         parameters = {i: [j] for i, j in cb_model.get_params().items()}
        #
        #     model_cv = GridSearchCV(cb_model, parameters, n_jobs=-1, scoring='roc_auc',
        #                             cv=folds,
        #                             verbose=1,
        #                             return_train_score=True)
        #
        #     model_cv.fit(X_train, y_train)
        #     print("Best roc auc score : ", model_cv.best_score_)
        #     print(model_cv.best_estimator_)
        #
        #     # cv results
        #     cv_results = pd.DataFrame(model_cv.cv_results_)
        #     return (model_cv, cv_results)
        #
        # def rf_classifier(folds=3,
        #                   param_grid={'bootstrap': [True, False],
        #                               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        #                               'max_features': ['auto', 'sqrt'],
        #                               'min_samples_leaf': [1, 2, 4],
        #                               'min_samples_split': [2, 5, 10],
        #                               'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}):
        #
        #     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
        #
        #     rf_model = RandomForestClassifier()
        #
        #     if optimize_hyperparameter == False:
        #         param_grid = {i: [j] for i, j in rf_model.get_params().items()}
        #
        #     model_cv = GridSearchCV(estimator=rf_model,
        #                             param_grid=param_grid,
        #                             scoring='roc_auc',
        #                             cv=folds,
        #                             verbose=1,
        #                             return_train_score=True)
        #
        #     model_cv.fit(X_train, y_train)
        #     print("Best roc auc score : ", model_cv.best_score_)
        #     print(model_cv.best_estimator_)
        #
        #     # cv results
        #     cv_results = pd.DataFrame(model_cv.cv_results_)
        #     return (model_cv, cv_results)
        print(type(ds))
        modelss = My_function(ds)
        import joblib
        import os
        path = 'static'
        PROJECT_ROOT_DIR = path
        IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "models_temp", "")
        import shutil
#        shutil.rmtree(IMAGES_PATH)
        os.makedirs(IMAGES_PATH, exist_ok=True)
        li_path = []
        for key in modelss.keys():
            joblib.dump(modelss[key], IMAGES_PATH + key + '.sav')
            patt = os.path.join(IMAGES_PATH, key + ".sav")
            li_path.append(patt)
            print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            print(patt)
            print(li_path)
        li_pp = str(li_path)
        print(li_pp, type(li_pp))
        # import csv
        # path = "static"
        # PROJECT_ROOT_DIR = path
        # CHAPTER_ID = "classification"
        # IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "modeldata", "")
        # # conf_path = os.path.join(PROJECT_ROOT_DIR, "images", "conf")
        # # print(conf_path)
        # print(IMAGES_PATH)
        # os.makedirs(IMAGES_PATH, exist_ok=True)
        # filename="data.csv"
        # name=os.path.join(IMAGES_PATH, filename)
        # f = open(name,'wb')
        # w = csv.DictWriter(f,modelss.keys())
        # w.writerow(modelss)
        # f.close()
        # # app_json = json.dumps(modelss)
        print("+++++++++++++++++++")
        # print(type(app_json))

        print(type(modelss))
        print(modelss)
        # models=list(models)

        # json_object = json.dumps(models)

        # for key in models.keys():
        #     joblib.dump(models[key], Project_Folder + r'/Models/'+key + '.sav')
        # mod=json.dumps(models)

        # print(mod)

        # mod=json.dumps(models)
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||vvv")
        print("ppp")
        # if 'Logistic Regression' in ds:
        #     print("log")
        #     models['logistic_regression'] = logistic_regression()
        #
        # if 'Decision Tree' in ds:
        #     print("dec")
        #     models['decision_tree'] = decision_tree()
        # if 'XGBoost' in ds:
        #     print("Xg")
        #     models['xgboost_classifier'] = xgboost_classifier()
        # if 'Gradient Boosting Machines' in ds:
        #
        #     print("Gbm")
        #     models['gbm_classifier'] = gbm_classifier()
        # if 'Light GBM' in ds:
        #     print("light")
        #     models['lgbm_classifier'] = lgbm_classifier()
        # if 'CATBOOST' in ds:
        #     print("Catboost")
        #     models['cb_classifier'] = cb_classifier()
        # if 'Random Forest' in ds:
        #     print("rf")
        #     models['rf_classifier'] = rf_classifier()
        print(
            "=====================================================================================================================")
        print(models)
        print(
            "========================================================================================================================")
        scoring_dict = dict()
        emptylist=[]
        from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
        for i in modelss.keys():
            empty_dic = dict()
            result_out = modelss[i][0].predict(X_test)
            print("avinash")
            # print(i)
            # result_out=list(result_out)
            # print(result_out)
            # print(result_out,type(result_out))
            # path = "static"
            # filepath = os.path.join(path, "resultss")
            # os.makedirs(filepath, exist_ok=True)
            # file = "result.csv"
            # file_path4 = os.path.join(filepath, file)
            # import csv
            # with open(file_path4, 'w') as f:
            #     write = csv.writer(f)
            #     write.writerow(result_out)
            # print("23")
            # # result_out.to_csv(os.path.join(filepath,"resultout.csv"))
            # print("23")
            # dic={"model_name":i,"result_out":result_out}
            # emptylist.append(dic)
            # print(dic)
            # print("avinash")
            empty_dic['log_loss'] = log_loss(y_test, result_out)
            empty_dic['roc_auc_score'] = roc_auc_score(y_test, result_out)
            empty_dic['accuracy_score'] = accuracy_score(y_test, result_out)
            scoring_dict[i] = empty_dic
            # val[i]={"name":scoring_dict}
            # val=val
            # print(scoring_dict)
            # print(type(scoring_dict))
        # p=scoring_dict
        # print(val)
        # emptylist.to_csv(file_path2)
        # print(emptylist,type(emptylist))
        #
        # # path="static"
        # path = "static"
        # print("1")
        #
        # filepath = os.path.join(path, "result")
        # print("2")
        # # shutil.rmtree(filepath)
        # print("3")
        # os.makedirs(filepath, exist_ok=True)
        # print("sasasasa")
        #
        # # textfile = "text.json"
        # file_path3 = os.path.join(filepath, "text.csv")
        # # with open(path, "w") as fs:
        #     print("in")
        #     json.dump(json_string, fs)
        #     # f.write(json_string)
        #     print("out")
        # print(file_path3)
        # keys = emptylist[0].keys()
        # import csv
        # with open(file_path3, 'w', newline='') as output_file:
        #     dict_writer = csv.DictWriter(output_file, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(emptylist)
        new_scoring_dict = pd.DataFrame(scoring_dict).T.to_dict()
        sorted_scoring_dict = dict()
        for key in new_scoring_dict.keys():
            if key == 'log_loss':
                sorted_scoring_dict['log_loss'] = dict(
                    sorted(new_scoring_dict['log_loss'].items(), key=operator.itemgetter(1), reverse=False))
            if key == 'accuracy_score':
                sorted_scoring_dict['accuracy_score'] = dict(
                    sorted(new_scoring_dict['accuracy_score'].items(), key=operator.itemgetter(1), reverse=True))
            if key == 'roc_auc_score':
                sorted_scoring_dict['roc_auc_score'] = dict(
                    sorted(new_scoring_dict['roc_auc_score'].items(), key=operator.itemgetter(1), reverse=True))
            sorted_scoring_dict
        df_rank = pd.DataFrame(sorted_scoring_dict)
        df_rank['log_loss_rank'] = df_rank['log_loss'].rank(method="dense", ascending=True)
        df_rank['roc_auc_score_rank'] = df_rank['roc_auc_score'].rank(method="dense", ascending=False)
        df_rank['accuracy_score_rank'] = df_rank['accuracy_score'].rank(method="dense", ascending=False)
        df_rank['accuracy_score_rank'] = df_rank['accuracy_score_rank'].apply(lambda x: 'rank' + str(int(x)))
        df_rank['roc_auc_score_rank'] = df_rank['roc_auc_score_rank'].apply(lambda x: 'rank' + str(int(x)))
        df_rank['log_loss_rank'] = df_rank['log_loss_rank'].apply(lambda x: 'rank' + str(int(x)))
        df_rank = df_rank.reset_index()
        import re
        updated_scoring_dict = dict()
        for i in ['log_loss_rank', 'roc_auc_score_rank', 'accuracy_score_rank']:
            temp_dict = (dict(zip(df_rank[i], df_rank['index'])))
            temp_dict.update(dict(zip(df_rank[i].apply(lambda x: x + 'value'), df_rank[re.sub('_rank', '', i)])))
            updated_scoring_dict[re.sub('_rank', '', i)] = temp_dict
        updated_scoring_dict
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(updated_scoring_dict)
        print(scoring_dict)
        s = [updated_scoring_dict]
        # s = [sorted_scoring_dict]
        con = json.dumps(s)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(con)
        print(type(con))
        print(project_name)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        current_date = datetime.datetime.now()
        Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
        print("ssss")
        # event={"name":"sai","res":"redd"}
        # print(event["res"])
        project_name = project_name
        con = con
        print(project_name, con)
        from app import models
        jobs_res = models.Sai.objects.all().delete()
        start_modeling_create = models.Sai(
            upload_id=Id_val,
            project_name=project_name,
            model_res=con,
            target_column=target_column,
            split_percentage=split_percentag,
            modells=ds,
            models_path=li_pp,
            target_variable=dd,
            # result=file_path3,
            xtest=file_path,
            ytest=file_path1
            # mod_data=app_json

        )
        start_modeling_create.save()
        print("after")
        print("after")

        res = {
            "status": "successs",
            "message": "details get is successsss",
            "data": [sorted_scoring_dict]
        }
        return JsonResponse(res)

    except Exception as e:
        res = {
            "status": "failed",
            "message": str(e)
        }
        return JsonResponse(res)


# def logistic_regression(params={"C": [0.01, 0.1, 1, 10, 100, 1000]},
#                         folds=KFold(n_splits=5, shuffle=True, random_state=4)
#                         ):
#     scaler = StandardScaler()
#
#     train_columns = X_train.columns
#     X_train_ = pd.DataFrame(data=scaler.fit_transform(X_train), columns=train_columns)
#     X_test_ = pd.DataFrame(data=scaler.fit_transform(X_test), columns=train_columns)
#     X_train_ = X_train_.fillna(0)
#     X_test_ = X_test_.fillna(0)
#
#     if optimize_hyperparameter == False:
#         params = {i: [j] for i, j in LogisticRegression().get_params().items()}
#
#     model_cv = GridSearchCV(estimator=LogisticRegression(),
#                             param_grid=params,
#                             scoring='roc_auc',
#                             cv=folds,
#                             n_jobs=-1,
#                             verbose=1,
#                             return_train_score=True)
#     # perform hyperparameter tuning
#     model_cv.fit(X_train_, y_train)
#     print('Best ROC AUC score: ', model_cv.best_score_)
#     # print the optimum value of hyperparameters
#     print('Best hyperparameters: ', model_cv.best_params_)
#
#     cv_results = pd.DataFrame(model_cv.cv_results_)
#     return (model_cv, cv_results)
#
#
# def decision_tree(param_grid={'max_depth': range(5, 15, 5), 'min_samples_leaf': range(50, 150, 50),
#                               'min_samples_split': range(50, 150, 50), }):
#     # Instantiate the grid search model
#     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
#
#     dtree = DecisionTreeClassifier()
#
#     if optimize_hyperparameter == False:
#         param_grid = {i: [j] for i, j in dtree.get_params().items()}
#
#     grid_search = GridSearchCV(estimator=dtree,
#                                param_grid=param_grid,
#                                scoring='roc_auc',
#                                cv=3,
#                                n_jobs=-1,
#                                verbose=1)
#
#     # Fit the grid search to the data
#     grid_search.fit(X_train, y_train)
#
#     print("Best roc auc score : ", grid_search.best_score_)
#     print(grid_search.best_estimator_)
#
#     cv_results = pd.DataFrame(grid_search.cv_results_)
#     return (grid_search, cv_results)
#
#
# def xgboost_classifier(folds=3,
#                        param_grid={'learning_rate': [0.2, 0.6],
#                                    'subsample': [0.3, 0.6, 0.9]}):
#     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
#
#     xgb_model = XGBClassifier(max_depth=2, n_estimators=200)
#
#     if optimize_hyperparameter == False:
#         param_grid = {i: [j] for i, j in xgb_model.get_params().items()}
#
#     model_cv = GridSearchCV(estimator=xgb_model,
#                             param_grid=param_grid,
#                             scoring='roc_auc',
#                             cv=folds,
#                             verbose=1,
#                             return_train_score=True)
#
#     model_cv.fit(X_train, y_train)
#     print("Best roc auc score : ", model_cv.best_score_)
#     print(model_cv.best_estimator_)
#
#     # cv results
#     cv_results = pd.DataFrame(model_cv.cv_results_)
#     return (model_cv, cv_results)
#
#
# def gbm_classifier(parameters={
#     "loss": ["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth": [3, 5, 8],
#     "max_features": ["log2", "sqrt"],
#     "criterion": ["friedman_mse", "mae"],
#     "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators": [10]
# }, folds=10):
#     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
#
#     gbm_model = GradientBoostingClassifier()
#     print("gbmgbmgmbmmbgm")
#     if optimize_hyperparameter == False:
#         parameters = {i: [j] for i, j in gbm_model.get_params().items()}
#     print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
#     model_cv = GridSearchCV(gbm_model, parameters, n_jobs=-1, scoring='roc_auc',
#                             cv=folds,
#                             verbose=1,
#                             return_train_score=True)
#
#     model_cv.fit(X_train, y_train)
#     print("Best roc auc score : ", model_cv.best_score_)
#     print(model_cv.best_estimator_)
#
#     # cv results
#     cv_results = pd.DataFrame(model_cv.cv_results_)
#     print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
#     print(cv_results)
#     print("/////////////////////////")
#     return (model_cv, cv_results)
#
#
# def lgbm_classifier(parameters={'learning_rate': [0.01], 'n_estimators': [8, 24],
#                                 'num_leaves': [6, 8, 12, 16], 'boosting_type': ['gbdt'],
#                                 'objective': ['binary'], 'seed': [500],
#                                 'colsample_bytree': [0.65, 0.75, 0.8],
#                                 'subsample': [0.7, 0.75], 'reg_alpha': [1, 2, 6],
#                                 'reg_lambda': [1, 2, 6]}, folds=10):
#     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
#
#     lgb_model = lgb.LGBMClassifier()
#
#     if optimize_hyperparameter == False:
#         parameters = {i: [j] for i, j in lgb_model.get_params().items()}
#
#     model_cv = GridSearchCV(lgb_model, parameters, n_jobs=-1, scoring='roc_auc',
#                             cv=folds,
#                             verbose=1,
#                             return_train_score=True)
#
#     model_cv.fit(X_train, y_train)
#     print("Best roc auc score : ", model_cv.best_score_)
#     print(model_cv.best_estimator_)
#
#     # cv results
#     cv_results = pd.DataFrame(model_cv.cv_results_)
#     return (model_cv, cv_results)
#
#
# def cb_classifier(parameters={'iterations': [500],
#                               'depth': [4, 5, 6],
#                               'loss_function': ['Logloss', 'CrossEntropy'],
#                               'l2_leaf_reg': np.logspace(-20, -19, 3),
#                               'leaf_estimation_iterations': [10],
#                               'logging_level': ['Silent'],
#                               'random_seed': [42]
#                               }, folds=10):
#     #     (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
#
#     cb_model = CatBoostClassifier()
#
#     if optimize_hyperparameter == False:
#         parameters = {i: [j] for i, j in cb_model.get_params().items()}
#
#     model_cv = GridSearchCV(cb_model, parameters, n_jobs=-1, scoring='roc_auc',
#                             cv=folds,
#                             verbose=1,
#                             return_train_score=True)
#
#     model_cv.fit(X_train, y_train)
#     print("Best roc auc score : ", model_cv.best_score_)
#     print(model_cv.best_estimator_)
#
#     # cv results
#     cv_results = pd.DataFrame(model_cv.cv_results_)
#     return (model_cv, cv_results)
#
#
# def rf_classifier(folds=3,
#                   param_grid={'bootstrap': [True, False],
#                               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#                               'max_features': ['auto', 'sqrt'],
#                               'min_samples_leaf': [1, 2, 4],
#                               'min_samples_split': [2, 5, 10],
#                               'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}):
#     #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test
#
#     rf_model = RandomForestClassifier()
#
#     if optimize_hyperparameter == False:
#         param_grid = {i: [j] for i, j in rf_model.get_params().items()}
#
#     model_cv = GridSearchCV(estimator=rf_model,
#                             param_grid=param_grid,
#                             scoring='roc_auc',
#                             cv=folds,
#                             verbose=1,
#                             return_train_score=True)
#
#     model_cv.fit(X_train, y_train)
#     print("Best roc auc score : ", model_cv.best_score_)
#     print(model_cv.best_estimator_)
#
#     # cv results
#     cv_results = pd.DataFrame(model_cv.cv_results_)
#     return (model_cv, cv_results)


@api_view(['GET', 'POST', ])
def model_res(request):
    try:

        operations_res = json.loads(serializers.serialize("json", models.Sai.objects.all()))
        print("++++++++++++++++++", operations_res)
        print(type(operations_res))
        a = operations_res[0]
        p = a["fields"]
        pa = p["model_res"]
        paa = ast.literal_eval(pa)
        print(type(paa))
        print(paa)
        operations = json.loads(serializers.serialize("json", models.Upload.objects.all()))
        aa = operations[0]
        q = aa["fields"]
        path = q["upload_file_path"]
        print(path)
        reading_csv = pd.read_csv(path)
        coloumn = reading_csv.columns
        lists = coloumn.tolist()
        json_str = json.dumps(lists)
        ress = json_str.strip('][').split(', ')
        item = {}
        res = []
        for i in ress:
            # Id=i
            i = i.strip('"')
            print(i)
            item["userId"] = i
            # item["body"]="sai prakash reddy"
            Val = {"userId": item["userId"]}
            res.append(Val)
            print(res)
        print(
            "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

        res = {
            "status": "success",
            "message": "details get is successsss",
            "data": paa, "summary": res
        }
        return JsonResponse(res)
    except Exception as e:
        res = {
            "status": "failed",
            "message": str(e)
        }
        return JsonResponse(res)


def mod_filter(conditions, df):
    new_dict = dict()
    for key in conditions.keys():
        if key != 'condition':
            if (df[key].dtype == ('int64') or df[key].dtype == ('float64')):
                new_dict[key] = conditions[key]
            else:
                new_dict[key + '_' + conditions[key][0]] = [1, '==']
        if key == 'condition':
            new_dict[key] = conditions[key]

    return new_dict


def filter_data(new_dict, X_test):
    print("filterfilter")
    print(new_dict)
    print(X_test)
    empty_df = pd.DataFrame()
    if new_dict['condition'] == 'AND':
        empty_df = copy.deepcopy(X_test)
        for key in new_dict.keys():

            if key != 'condition':
                print(new_dict[key][1])
                if new_dict[key][1] == '=':
                    empty_df = empty_df[empty_df[key] == new_dict[key][0]]
                if new_dict[key][1] == '<=':
                    print("fixing")
                    print(new_dict[key][0])
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
    print("=====================")
    print(empty_df)

    return empty_df


def My_functions(ds):
    modelss = dict()
    if 'Logistic Regression' in ds:
        print("log")
        modelss['logistic_regression'] = logistic_regression()

    if 'Decision Tree' in ds:
        print("dec")
        modelss['decision_tree'] = decision_tree()
    if 'XGBoost' in ds:
        print("Xg")
        modelss['xgboost_classifier'] = xgboost_classifier()
    if 'Gradient Boosting Machines' in ds:
        print("Gbm")
        modelss['gbm_classifier'] = gbm_classifier()
    if 'Light GBM' in ds:
        print("light")
        modelss['lgbm_classifier'] = lgbm_classifier()
    if 'CATBOOST' in ds:
        print("Catboost")
        modelss['cb_classifier'] = cb_classifier()
    if 'Random Forest' in ds:
        print("rf")
        modelss['rf_classifier'] = rf_classifier()
    # print(modelss)

    return modelss

#
# @api_view(['GET', 'POST', ])
# def model_evaluation(request):
#     try:
#         request_json = json.loads(request.body)
#         print("++++++++++++++++++", request_json)
#         # j=request_json
#         print(type(request_json))
#         j = request_json["InputFileds"]
#         print(j)
#         print("==============================================")
#         p = request_json["modelevaluated"]
#         print(p)
#         print("==========================================")
#         # modelList=p["modelList"]
#         modelList = p
#         print(modelList)
#         rr = []
#         rr.append(modelList)
#         print(rr)
#
#         dict_ = dict(zip(range(len(j)), j))
#         di = dict_
#         print(di)
#         temp_dict = dict()
#         print("[][]+++++++++++++++++++++++++++[][]")
#         operations_res = json.loads(serializers.serialize("json", models.Sai.objects.all()))
#         # print("++++++++++++++++++", operations_res)
#         print(operations_res)
#         a = operations_res[0]
#
#         print(type(a))
#         p = a["fields"]
#
#         target_column = p["target_column"]
#         print(target_column)
#         split_percentage = p["split_percentage"]
#         modells = p["modells"]
#
#         # res = ast.literal_eval(modells)
#
#         # res=modells.strip('\"')
#         # res=modells[1:-1]
#         # ress=res[1:-1]
#         # res=literal_eval(modells)
#
#         print("================")
#         res = modells.strip('][').split(', ')
#         rem = []
#         for ress in res:
#             print(ress)
#
#             ress = ress[1:-1]
#             print(ress)
#             rem.append(ress)
#         print(type(rem))
#         ds = rem
#         print(ds)
#         # modelss = My_function(ds)
#         print("{][][][][][][][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[")
#         # print(modelss)
#         # print(res[0])
#
#         print("[][]+++++++++++++++++++++++++++[][]")
#
#         operations_res = json.loads(serializers.serialize("json", models.Upload.objects.all()))
#         res = []
#         a = operations_res[0]
#         p = a["fields"]
#         print(p)
#         project_name = p["project_name"]
#         print(project_name)
#         path = p["upload_file_path"]
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print(path)
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         df = pd.read_csv(path)
#         df_cat = df.select_dtypes(exclude=['float64', 'int64'])
#         df_int = df.select_dtypes(include=['float64', 'int64'])
#         df_cat = df_cat.fillna('NA')
#         df_int = df_int.fillna(0)
#         df = pd.concat([df_cat, df_int], axis=1)
#         y = df[target_column]
#         print(y, "yyyyyyyyyyyy")
#         X = df.drop(target_column, axis=1)
#         print(X, "xxxxxxxx")
#         X = pd.get_dummies(X)
#         print(X, "Xy xy")
#         split_percentage = split_percentage
#         split_percentage = int(split_percentage) / 100
#         print(type(split_percentage))
#         print(split_percentage, "split percentage")
#         global X_train
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=None)
#         optimize_hyperparameter = False
#
#         print(X_test, X_train, y_train, y_test)
#         import joblib
#         import os
#         path = 'static'
#         PROJECT_ROOT_DIR = path
#         IMAGES_PATHh = os.path.join(PROJECT_ROOT_DIR, "models_temp", "")
#
#         import pickle
#         import re
#
#         model_names = os.listdir(IMAGES_PATHh)
#         modelss = dict()
#         for item in model_names:
#             modelss[item[:-4]] = joblib.load(IMAGES_PATHh + item)
#         print("PPPPPPPPPPPPPPP", type(modelss))
#         print(modelss)
#
#         print("saisas")
#         # ds=['Decision Tree']
#         # modelss = My_function(ds)
#         print("{][][][][][][][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[")
#
#         print(modelss)
#
#         print("[][]+++++++++++++++++++++++++++[][]")
#
#         for value in dict_.values():
#             print("{}{}{}{}{}")
#             print(value)
#             print(df_int, "=================================================================")
#
#             if value['selectVariable'] in df_int.columns:
#                 print("[][][][][][]")
#                 temp_dict[value['selectVariable']] = [float(value['evaluationValue'])]
#                 temp_dict[value['selectVariable']].append(value['sampling'])
#                 temp_dict[value['selectVariable']].append(value['binaryOperation'])
#                 temp_dict['condition'] = value['binaryOperation']
#                 # print(temp_dict,"=========")
#
#             if value['selectVariable'] in df_cat.columns:
#                 temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])] = [1]
#                 temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(value['sampling'])
#                 temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(
#                     value['binaryOperation'])
#                 temp_dict['condition'] = value['binaryOperation']
#         print(temp_dict, "==================")
#         print("+++++++++++++++++++++++++++++++++++++++++++")
#
#         # temp_dict['condition']==
#         print("X_test", X_test)
#         if len(temp_dict) > 0:
#             X_filter = filter_data(temp_dict, X_test)
#         else:
#             X_filter = X_test
#         print(X_filter.shape)
#         print("X_filter", X_filter, "====")
#         y_filter = y_test[y_test.index.isin(X_filter.index)]
#         print("y_filter", y_filter, "\\\\\\\\\\\\\\\\+++++++++++++++++++++++")
#         filtered_predictions = dict()
#         filtered_predictions_prob = dict()
#         for key in modelss.keys():
#             print("mounesh")
#             filtered_predictions[key] = modelss[key][0].predict(X_filter)
#             filtered_predictions_prob[key] = modelss[key][0].predict_proba(X_filter)
#         print(filtered_predictions)
#         print(filtered_predictions_prob)
#         print("done")
#         import scikitplot as skplt
#         # import scikitplot.plotters as skplt
#         # import scikitplot.metrics.plot_roc as skplt
#         import matplotlib.pyplot as plt
#
#         path = "static"
#         PROJECT_ROOT_DIR = path
#         CHAPTER_ID = "classification"
#         IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "sai", CHAPTER_ID, "")
#         # conf_path = os.path.join(PROJECT_ROOT_DIR, "images", "conf")
#         # print(conf_path)
#         print(IMAGES_PATH)
#         os.makedirs(IMAGES_PATH, exist_ok=True)
#         # os.makedirs(conf_path, exist_ok=True)
#         # s = os.path.join(IMAGES_PATH, "\\")
#         # s=IMAGES_PATH+s
#         # print(s)
#
#         for key, value in filtered_predictions_prob.items():
#             print("++++++")
#             skplt.metrics.plot_roc_curve(y_filter.values, value)
#             print("+++++++++++++++")
#             plt.savefig(IMAGES_PATH + 'roc.png')
#
#         temp_fig_path = 'roc.png'
#         imagepath = os.path.join(IMAGES_PATH, temp_fig_path)
#         # imagepath = IMAGES_PATH + temp_fig_path
#         print(imagepath, "temp_fig_path")
#         print("dodododododododo")
#         path = "static"
#         PROJECT_ROOT_DIR = path
#         conf_path = os.path.join(PROJECT_ROOT_DIR, "images", "conf", "")
#         print("000000000000")
#         print(conf_path)
#         print("0000000000000")
#         os.makedirs(conf_path, exist_ok=True)
#         import seaborn as sns
#         from sklearn.metrics import confusion_matrix
#         for key, value in filtered_predictions.items():
#             print("++++++")
#
#             cm = confusion_matrix(y_filter.values, value)
#             f = sns.heatmap(cm, annot=True, fmt='d')
#             plt.savefig(conf_path + 'confusion.png')
#         temp_path = 'confusion.png'
#         # confimage = conf_path + temp_path
#         confimage = os.path.join(conf_path, temp_path)
#         print(confimage, "confusion path")
#         ppp = "http:\\\localhost:8000"
#         rc = os.path.join(ppp, imagepath)
#         co = os.path.join(ppp, confimage)
#
#         print(rc, type(rc))
#         print(co, type(co))
#         # rc = imagepath
#         # co = confimage
#         # co="http://localhost:8000/static/images/conf/confusion.png"
#         # rc="http://localhost:8000/static/images/classification/roc.png"
#         # jobs_res = models.Image.objects.all().delete()
#         # current_date = datetime.datetime.now()
#         # Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
#         # image_create = models.Image(
#         #     image_id=Id_val,
#         #     file_path_roc=rc,
#         #     file_path_con=co
#         #
#         #
#         # )
#         # image_create.save()
#         # print("after")
#         a_dict = {}
#
#         for variable in ["rc", "co"]:
#             a_dict[variable] = eval(variable)
#         print(a_dict, "{}{}{}{}{}{")
#         rem = {
#             "roc_image": "http://localhost:8000/static/images/classification/rf_classifier.png",
#             "con_image": "http://localhost:8000/static/images/conf/rf_classifier.png"}
#         imagess = {"roc_image": rc,
#                    "con_image": co}
#         images = dict()
#         for k, v in imagess.items():
#             images[k] = (str(v.replace('\\', '/')))
#
#         # images
#         print(images)
#         jobs_res = models.Image.objects.all().delete()
#         current_date = datetime.datetime.now()
#         Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
#         image_create = models.Image(
#             image_id=Id_val,
#             file_path_roc=rc,
#             file_path_con=co
#
#         )
#         image_create.save()
#         print("after")
#         print("saisasasa")
#
#         # import numpy as np
#         from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#         for key, value in filtered_predictions.items():
#             y_true = y_filter
#             y_pred = value
#             j = precision_recall_fscore_support(y_true, y_pred, average='macro')
#             precision_, recall_, f1_score_, accuracy_ = j[0], j[1], j[2], accuracy_score(y_true, y_pred)
#             dictionary_for_confusion = {'precision': precision_, 'recall': recall_, 'f1_score': f1_score_,
#                                         'accuracy': accuracy_}
#         print(dictionary_for_confusion, "dictionary_for_confusion")
#         res = {
#             "status": "success",
#             "message": "details get is successsss",
#             "data": dictionary_for_confusion
#
#         }
#         return JsonResponse(res)
#     except Exception as e:
#         res = {
#             "status": "failed",
#             "message": str(e)
#         }
#         return JsonResponse(res)


@api_view(['GET', 'POST', ])
def model_evaluation(request):
    try:
        request_json = json.loads(request.body)
        print("++++++++++++++++++", request_json)
        # j=request_json
        print(type(request_json))
        j = request_json["InputFileds"]
        print(j)
        print("==============================================")
        p = request_json["modelevaluated"]
        print(p)
        print("==========================================")
        # modelList=p["modelList"]
        modelList = p
        print(modelList)
        rr = []
        rr.append(modelList)
        print(rr)

        dict_ = dict(zip(range(len(j)), j))
        di = dict_
        print(di)
        temp_dict = dict()
        print("[][]+++++++++++++++++++++++++++[][]")
        operations_res = json.loads(serializers.serialize("json", models.Sai.objects.all()))
        # print("++++++++++++++++++", operations_res)
        print(operations_res)
        a = operations_res[0]

        print(type(a))
        p = a["fields"]

        target_column = p["target_column"]
        print(target_column)
        split_percentage = p["split_percentage"]
        modells = p["modells"]

        # res = ast.literal_eval(modells)

        # res=modells.strip('\"')
        # res=modells[1:-1]
        # ress=res[1:-1]
        # res=literal_eval(modells)

        print("================")
        res = modells.strip('][').split(', ')
        rem = []
        for ress in res:
            print(ress)

            ress = ress[1:-1]
            print(ress)
            rem.append(ress)
        print(type(rem))
        ds = rem
        print(ds)
        # modelss = My_function(ds)
        print("{][][][][][][][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[")
        # print(modelss)
        # print(res[0])

        print("[][]+++++++++++++++++++++++++++[][]")

        operations_res = json.loads(serializers.serialize("json", models.Upload.objects.all()))
        res = []
        a = operations_res[0]
        p = a["fields"]
        print(p)
        project_name = p["project_name"]
        print(project_name)
        path = p["upload_file_path"]
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(path)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        df = pd.read_csv(path)
        df_cat = df.select_dtypes(exclude=['float64', 'int64'])
        df_int = df.select_dtypes(include=['float64', 'int64'])
        df_cat = df_cat.fillna('NA')
        df_int = df_int.fillna(0)
        df = pd.concat([df_cat, df_int], axis=1)
        y = df[target_column]
        print(y, "yyyyyyyyyyyy")
        X = df.drop(target_column, axis=1)
        print(X, "xxxxxxxx")
        X = pd.get_dummies(X)
        print(X, "Xy xy")
        split_percentage = split_percentage
        split_percentage = int(split_percentage) / 100
        print(type(split_percentage))
        print(split_percentage, "split percentage")
        global X_train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=None)
        optimize_hyperparameter = False

        print(X_test, X_train, y_train, y_test)
        import joblib
        import os
        path = 'static'
        PROJECT_ROOT_DIR = path
        IMAGES_PATHh = os.path.join(PROJECT_ROOT_DIR, "models_temp", "")

        import pickle
        import re

        model_names = os.listdir(IMAGES_PATHh)
        modelss = dict()
        for item in model_names:
            modelss[item[:-4]] = joblib.load(IMAGES_PATHh + item)
        print("PPPPPPPPPPPPPPP",type(modelss))
        print(modelss)


        print("saisas")
                # ds=['Decision Tree']
        # modelss = My_function(ds)
        print("{][][][][][][][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[")

        print(modelss)

        print("[][]+++++++++++++++++++++++++++[][]")

        for value in dict_.values():
            print("{}{}{}{}{}")
            print(value)
            print(df_int, "=================================================================")

            if value['selectVariable'] in df_int.columns:
                print("[][][][][][]")
                temp_dict[value['selectVariable']] = [float(value['evaluationValue'])]
                temp_dict[value['selectVariable']].append(value['sampling'])
                temp_dict[value['selectVariable']].append(value['binaryOperation'])
                temp_dict['condition'] = value['binaryOperation']
                # print(temp_dict,"=========")

            if value['selectVariable'] in df_cat.columns:
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])] = [1]
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(value['sampling'])
                temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(
                    value['binaryOperation'])
                temp_dict['condition'] = value['binaryOperation']
        print(temp_dict, "==================")
        print("+++++++++++++++++++++++++++++++++++++++++++")

        # temp_dict['condition']==
        print("X_test", X_test.columns)
        print(X_test)
        if len(temp_dict) > 0:
            X_filter = filter_data(temp_dict, X_test)
        else:
            X_filter = X_test
            print("elseelse")
        print(X_filter)
        print(X_filter.shape)
        print("X_filter", X_filter, "====","avinash")
        y_filter = y_test[y_test.index.isin(X_filter.index)]
        print("y_filter", y_filter, "\\\\\\\\\\\\\\\\+++++++++++++++++++++++")
        filtered_predictions = dict()
        filtered_predictions_prob = dict()
        for key in modelss.keys():
            filtered_predictions[key] = modelss[key][0].predict(X_filter)
            filtered_predictions_prob[key] = modelss[key][0].predict_proba(X_filter)
        print(filtered_predictions)
        print(filtered_predictions_prob)
        print("done")
        import scikitplot as skplt
        # import scikitplot.plotters as skplt
        # import scikitplot.metrics.plot_roc as skplt
        import matplotlib.pyplot as plt

        path = "static"
        PROJECT_ROOT_DIR = path
        CHAPTER_ID = "classification"
        IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "sa", CHAPTER_ID, "")
        # conf_path = os.path.join(PROJECT_ROOT_DIR, "images", "conf")
        # print(conf_path)
        print(IMAGES_PATH)
        # os.rmdir(IMAGES_PATH)
        # sleep(15)
        # shutil.rmtree(IMAGES_PATH)
        os.makedirs(IMAGES_PATH, exist_ok=True)

        # os.makedirs(conf_path, exist_ok=True)
        # s = os.path.join(IMAGES_PATH, "\\")
        # s=IMAGES_PATH+s
        # print(s)
        plt.clf()
        current_date = datetime.datetime.now()
        Id_val = current_date.strftime("%Y%m%d%H%M%S")
        roc=str(Id_val)+"."+"png"
        print("=================")
        print("========")
        print("===")
        print(roc)
        for key, value in filtered_predictions_prob.items():
            print("++++++")
            skplt.metrics.plot_roc_curve(y_filter.values, value)
            print("+++++++++++++++")
            plt.savefig(IMAGES_PATH + roc)
            # plt.show()
            plt.close()
            # plt.show()

        temp_fig_path = 'roc.png'
        imagepath = os.path.join(IMAGES_PATH, roc)
        # imagepath = IMAGES_PATH + temp_fig_path
        print(imagepath, "temp_fig_path")
        print("dodododododododo")
        path = "static"
        PROJECT_ROOT_DIR = path
        conf_path = os.path.join(PROJECT_ROOT_DIR, "images","sai","")
        print("000000000000")
        print(conf_path)
        print("0000000000000")
        os.makedirs(conf_path, exist_ok=True)
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        plt.clf()
        print("sai")
        current_date = datetime.datetime.now()
        Id_val = current_date.strftime("%Y%m%d%H%M%S")
        con = str(Id_val) + "." + "png"
        for key, value in filtered_predictions.items():
            print("++++++")

            cm = confusion_matrix(y_filter.values, value)
            f = sns.heatmap(cm, annot=True, fmt='d')
            plt.savefig(conf_path + con)
        # plt.show()
        # sleep(2)
            plt.close()
            # plt.show()
        temp_path = 'confusion.png'
        # confimage = conf_path + temp_path
        confimage = os.path.join(conf_path, con)
        print(confimage, "confusion path")
        ppp = "https:\\\edge-detect-v2.herokuapp.com"
        rc = os.path.join(ppp, imagepath)
        co = os.path.join(ppp, confimage)

        print(rc, type(rc))
        print(co, type(co))
        # rc = imagepath
        # co = confimage
        # co="http://localhost:8000/static/images/conf/confusion.png"
        # rc="http://localhost:8000/static/images/classification/roc.png"
        # jobs_res = models.Image.objects.all().delete()
        # current_date = datetime.datetime.now()
        # Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
        # image_create = models.Image(
        #     image_id=Id_val,
        #     file_path_roc=rc,
        #     file_path_con=co
        #
        #
        # )
        # image_create.save()
        # print("after")
        a_dict = {}

        for variable in ["rc", "co"]:
            a_dict[variable] = eval(variable)
        print(a_dict, "{}{}{}{}{}{")
        rem = {
            "roc_image": "http://localhost:8000/static/images/classification/rf_classifier.png",
            "con_image": "http://localhost:8000/static/images/conf/rf_classifier.png"}
        imagess = {"roc_image": rc,
                   "con_image": co}
        images = dict()
        for k, v in imagess.items():
            images[k] = (str(v.replace('\\', '/')))

        # images

        # import numpy as np
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        for key, value in filtered_predictions.items():
            y_true = y_filter
            y_pred = value
            j = precision_recall_fscore_support(y_true, y_pred, average='macro')
            precision_, recall_, f1_score_, accuracy_ = j[0], j[1], j[2], accuracy_score(y_true, y_pred)
            dictionary_for_confusion = {'precision': precision_, 'recall': recall_, 'f1_score': f1_score_,
                                        'accuracy': accuracy_}
        print(dictionary_for_confusion, "dictionary_for_confusion")
        res = dict()
        for key in dictionary_for_confusion:
            # rounding to K using round()
            res[key] = round(dictionary_for_confusion[key], 3)
        print(res)
        print(images)
        jobs_res = models.Image.objects.all().delete()
        current_date = datetime.datetime.now()
        Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
        image_create = models.Image(
            image_id=Id_val,
            file_path_roc=rc,
            file_path_con=co,
            data=dictionary_for_confusion

        )
        image_create.save()
        print("after")

        res = {
            "status": "success",
            "message": "details get is successsss",
            "data": res

        }
        return JsonResponse(res)
    except Exception as e:
        res = {
            "status": "failed",
            "message": str(e)
        }
        return JsonResponse(res)
import sklearn
import re
import copy
import pickle
import json
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



api_view(['GET', 'POST', ])
def finalimagepath(request):
    try:
        operations_res = json.loads(serializers.serialize("json", models.Segment.objects.all()))
        print(operations_res)
        fields = []
        for i in operations_res:
            print("00.1")
            segment = "segment" + str(len(fields))
            l = i["fields"]
            m = l["rule_data"]

            res = ast.literal_eval(m)
            p = res["InputFileds"]
            pp = p[0]
            mm = res["modelevaluated"]
            pp.update([('modelevaluated', mm)])
            del pp['id']
            print(pp, "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # fields.append(pp)

            segment_data = ast.literal_eval(m)
            # seg = {"InputFileds": fields}

            for item in segment_data.get("InputFileds"):
                del item["id"]
                item["modelevaluated"] = segment_data.get("modelevaluated")
            del segment_data["modelevaluated"]
            fields.append({segment: segment_data})

            operations_res = json.loads(serializers.serialize("json", models.Rules.objects.all()))
            print(operations_res)
            for i in operations_res:
                p = i["fields"]
                l = p["rule_data"]
                res = ast.literal_eval(l)
                print(type(res))
            rules = {'Rules': res}

        res = {"rules": rules, "segment": fields}

        json_string = json.dumps(res)

        path = "static"
        import os
        filepath = os.path.join(path, "seg")
#        shutil.rmtree(filepath)
        print("000000000000")
        print(filepath)
        print("0000000000000")
        os.makedirs(filepath, exist_ok=True)
        file = "segment_rules.json"
        file_path = os.path.join(filepath, file)

        with open(file_path, 'w') as fs:
            json.dump(json_string, fs)
        operations = json.loads(serializers.serialize("json", models.Upload.objects.all()))
        res = []
        print(operations)
        print(type(operations))
        a = operations[0]
        p = a["fields"]
        filepat = p["upload_file_path"]
        print(filepat)
        operations_res = json.loads(serializers.serialize("json", models.Sai.objects.all()))
        i=operations_res[0]
        print(i)
        ii=i["fields"]
        rank=ii["model_res"]
        d = ast.literal_eval(rank)
        dd=d[0]
        rank1= dd["accuracy_score"]
        rank1=rank1["rank1"]
        base_model=rank1+".sav"
        print("ramram")
        print(base_model)
        print(rank1)
        # result_out=ii["result"]
        X_test=ii["xtest"]
        y_test=ii["ytest"]

        # name_records = pd.read_csv(result_out)
        # name_records = name_records.to_dict('records')
        # print(name_records)
        # for i in name_records:
        #     print(i)
        #     if i["model_name"] == rank1:
        #         print("sai")
        #         y_pred = i["result_out"]
        #         print(y_pred)
        #     else:
        #         continue
        # print(y_pred)


        # starting

        # df = pd.read_csv(path)  # Need to read the original data from the folder.
        # X_test = pd.read_csv(X_test)
        # X_test = X_test.iloc[:, 1:]
        # y_test=pd.read_csv(y_test)
        #
        # y_test = y_test.iloc[:, -1:]
        # X_test['y_actual'] = y_test
        # X_test['y_pred'] = result_out
        import os
        path="static"
        pathh=os.path.join(path,"seg")
        pathhh=os.path.join(pathh,"segment_rules.json")

        print("1.1")
        f = open(pathhh)
        print("1.2")
        seg_rules_dict = json.load(f)
        print("1.4")
        print(seg_rules_dict,type(seg_rules_dict))
        d = ast.literal_eval(seg_rules_dict)
        # f = open('Segment_Rules.json')
        # seg_rules_dict = json.load(f)

        # In[209]:

        # Creating a dictionary for segment and model names
        temp_seg_dict = d["segment"]

        print("1.4.11")
        seg_models = {}
        print("1.4.1")
        for i in range(len(temp_seg_dict)):
            print("1.5")
            temp = temp_seg_dict[i]['segment' + str(i)]['InputFileds'][0]['modelevaluated']
            seg_models.update({'segment' + str(i): temp})

        # In[210]:

        # Loading the models
        import joblib
        import os
        seg_models_dic = {}
        for i in range(len(temp_seg_dict)):
            print("1.6")
            filename = seg_models['segment' + str(i)] + '.sav'
            print(filename)
            path = os.path.join("static", "models_temp", filename)
            loaded_model = joblib.load(open(path, 'rb'))
            print("1.7")
            # loaded_model = joblib.load(open(filename, 'rb'))
            model_name = seg_models['segment' + str(i)]
            seg_models_dic.update({model_name: loaded_model})
        # Loading base model

        filename = base_model
        path = os.path.join("static", "models_temp", filename)
        base_model = joblib.load(open(path, 'rb'))
        print("1.4")

        # In[ ]:

        df = pd.read_csv(filepat)  # Need to read the original data from the folder.
        print("1.8")
        df_cat_ = df.select_dtypes(exclude=['float64', 'int64'])
        df_int_ = df.select_dtypes(include=['float64', 'int64'])
        df_cat_.fillna('NA')
        df_int_.fillna(0)
        X_test = pd.read_csv( X_test)
        X_test = X_test.iloc[:, 1:]
        y_test = pd.read_csv(y_test)
        y_test = y_test.iloc[:, -1:]
        X_test['y_pred'] = base_model[0].predict(X_test)
        X_test['y_actual'] = y_test  # predicted values of best model
        print("1.9")

        # In[212]:

        # Functions to filter Data, Segmentation,Rules:

        # def filter_data(new_dict, df):
        #     print("1.9.1")
        #     empty_df = pd.DataFrame()
        #     print("1.1.1.1.1.1")
        #     if new_dict['condition'] == 'AND':
        #         empty_df = copy.deepcopy(df)
        #         for key in new_dict.keys():
        #             print("1.9.1.1.1")
        #             if key != 'condition':
        #                 if new_dict[key][1] == '=':
        #                     empty_df = empty_df[empty_df[key] == new_dict[key][0]]
        #                 if new_dict[key][1] == '<=':
        #                     empty_df = empty_df[empty_df[key] <= new_dict[key][0]]
        #                 if new_dict[key][1] == '>=':
        #                     empty_df = empty_df[empty_df[key] >= new_dict[key][0]]
        #                 if new_dict[key][1] == '<':
        #                     empty_df = empty_df[empty_df[key] < new_dict[key][0]]
        #                 if new_dict[key][1] == '>':
        #                     empty_df = empty_df[empty_df[key] > new_dict[key][0]]
        #
        #     if new_dict['condition'] == 'OR':
        #         print("1.9.1.1")
        #         for key in new_dict.keys():
        #             if key != 'condition':
        #                 if new_dict[key][1] == '=':
        #                     empty_df = pd.concat([empty_df, X_test[X_test[key] == new_dict[key][0]]], axis=0)
        #                 if new_dict[key][1] == '<=':
        #                     empty_df = pd.concat([empty_df, X_test[X_test[key] <= new_dict[key][0]]], axis=0)
        #                 if new_dict[key][1] == '>=':
        #                     empty_df = pd.concat([empty_df, X_test[X_test[key] >= new_dict[key][0]]], axis=0)
        #                 if new_dict[key][1] == '<':
        #                     empty_df = pd.concat([empty_df, X_test[X_test[key] < new_dict[key][0]]], axis=0)
        #                 if new_dict[key][1] == '>':
        #                     empty_df = pd.concat([empty_df, X_test[X_test[key] > new_dict[key][0]]], axis=0)
        #
        #     return empty_df

        def SegmentModelling_validationSet(X_test, seg_models_dic, temp_seg_dict, seg_models, df_int_, df_cat_):

            for i in range(len(temp_seg_dict)):
                print("1.9.2")
                dict_x = {}
                x = temp_seg_dict[i]['segment' + str(i)]['InputFileds'][0]
                del x['modelevaluated']
                dict_x.update({1: x})
                temp_dict = dict()

                for value in dict_x.values():
                    print("1.9.2.1")
                    print(value,type(value))
                    if value['selectVariable'] in df_int_.columns:
                        print("1.91.1")
                        temp_dict[value['selectVariable']] = [float(value['evaluationValue'])]
                        temp_dict[value['selectVariable']].append(value['sampling'])
                        temp_dict[value['selectVariable']].append(value['binaryOperation'])
                        temp_dict['condition'] = value['binaryOperation']
                    if value['selectVariable'] in df_cat_.columns:
                        print("1.1.1.1.1.1.1.1")
                        temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])] = [1]
                        temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(
                            value['sampling'])
                        temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(
                            value['binaryOperation'])
                        temp_dict['condition'] = value['binaryOperation']
                        print(value)

                X_filter = filter_data(temp_dict, X_test)
                print(X_test.shape,type(X_test.shape))
                rows = X_test.shape[0]
                print(X_filter.shape, type(X_filter.shape))
                if X_filter.shape[0] > 0 & X_filter.shape[0] < rows:
                    print("2.2.2.2.2.2")
                    X_test = pd.concat([X_test, X_filter]).drop_duplicates(keep=False)
                    model_name = seg_models['segment' + str(i)]
                    model = seg_models_dic[model_name]
                    actual = X_filter['y_actual']
                    X_filter = X_filter.drop(['y_pred'], axis=1)
                    X_filter = X_filter.drop(['y_actual'], axis=1)
                    y_filter = model[0].predict(X_filter)
                    X_filter['y_actual'] = actual
                    X_filter['y_pred'] = y_filter
                    X_test = X_test.append(X_filter)
                    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            print(X_test)
            return X_test

        def ApplyingRules(temp_rules_dict, modeled_df, df_int_, df_cat_):
            print(temp_rules_dict)
            print("121212121212121212")

            for i in range(len(temp_rules_dict)):
                dict_x = {}
                temp = temp_rules_dict['Rules'][i]
                dict_x = {1: temp}
                temp_dict = dict()
                for value in dict_x.values():
                    print("value")
                    if value['selectVariable'] in df_int_.columns:
                        temp_dict[value['selectVariable']] = [float(value['evaluationValue'])]
                        temp_dict[value['selectVariable']].append(value['selectOperator'])
                        temp_dict['condition'] = value['selectOperation']
                    if value['selectVariable'] in df_cat_.columns:
                        temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])] = [1]
                        temp_dict[value['selectVariable'] + '_' + str(value['evaluationValue'])].append(
                            value['selectOperator'])
                        temp_dict['condition'] = value['selectOperation']
                print("================================================================////////")
                print(temp_dict)
                X_filter = filter_data(temp_dict, modeled_df)
                modeled_df = pd.concat([modeled_df, X_filter]).drop_duplicates(keep=False)
                X_filter['y_pred'] = temp_rules_dict['Rules'][i]['selectedVariable']
                modeled_df = modeled_df.append(X_filter)
                # print(modeled_df)

                print("value1")

            return modeled_df

        # In[215]:

        results = SegmentModelling_validationSet(X_test, seg_models_dic, temp_seg_dict, seg_models, df_int_, df_cat_)
        print(results)
        print("sai")

        # In[217]:

        # temp_rules_dict = seg_rules_dict
        # print(temp_seg_dict,type(temp_seg_dict))
        temp_rules_dict = d["rules"]


        # In[219]:

        final_df = ApplyingRules(temp_rules_dict, results, df_int_, df_cat_)
        print("656565656565656")

        # In[221]:

        y_pred = list(final_df['y_pred'])
        # y_pred = list(final_df['y_pred'].astype(int))
        y_actual = list(final_df['y_actual'])
        # y_actual = list(final_df['y_actual'].astype(int))
        # y_pred=int(y_pred)
        # y_actual=int(y_actual)
        print(y_pred)
        print("=================")
        print(y_actual)

        # In[222]:

        from sklearn.metrics import confusion_matrix

        # Generate the confusion matrix
        print(y_actual)
        print("============")
        print(y_pred)
        cf_matrix = confusion_matrix(y_actual, y_pred)
        print("till here")

        # In[223]:

        import seaborn as sns
        plt.clf()

        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='d')
        # f = sns.heatmap(cm, annot=True, fmt='d')

        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])

        ## Display the visualization of the Confusion Matrix.
        # plt.show()
        path = "static"
        PROJECT_ROOT_DIR = path
        CHAPTER_ID = "finalimage"
        IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, CHAPTER_ID, "")
        # conf_path = os.path.join(PROJECT_ROOT_DIR, "images", "conf")
        # print(conf_path)
        print(IMAGES_PATH)
        os.makedirs(IMAGES_PATH, exist_ok=True)
        plt.savefig(IMAGES_PATH +'_confusion.png')
        plt.close()
        name="_confusion.png"
        finalimagepath=os.path.join(IMAGES_PATH,name)
        print(finalimagepath)
        ppp = "https:\\\edge-detect-v2.herokuapp.com"
        rc = os.path.join(ppp, finalimagepath)
        print(rc)
        rm=str(rc.replace('\\', '/'))
        print(rm)
        plt.close()


        # In[204]:

        from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

        # In[205]:

        empty_dic = dict()
        empty_dic['log_loss'] = log_loss(y_actual, y_pred)
        empty_dic['roc_auc_score'] = roc_auc_score(y_actual, y_pred)
        empty_dic['accuracy_score'] = accuracy_score(y_actual, y_pred)

        # In[206]:

        empty_dic
        print(empty_dic)
        res = dict()
        for key in empty_dic:
            # rounding to K using round()
            res[key] = round(empty_dic[key], 3)
        print(res)
        reee=[res]

        rem = {
            "con_image": rm}
        res = {"status": "success", "message": "upload created successfully", "image": rem,"scores":reee}
        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)

@api_view(['GET', 'POST', ])
def imagepath(request):
    try:
        rem = {
            "roc_image": "http://localhost:8000/static/images/classification/roc.png",
            "con_image": "http://localhost:8000/static/images/conf/confusion.png"}
        res = {"status": "success", "message": "upload created successfully", "image": rem}
        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


@api_view(['GET', 'POST', ])
def image_eva(request):
    try:
        sleep(10)
        operations_res = json.loads(serializers.serialize("json", models.Image.objects.all()))
        print(operations_res)
        im = operations_res[0]
        ima = im["fields"]
        print(ima)
        data=ima["data"]
        print(data,type(data))

        d=ast.literal_eval(data)
        res = dict()
        for key in d:
            # rounding to K using round()
            res[key] = round(d[key], 3)
        print(res)
        finaldata=[res]
        images = dict()
        for k, v in ima.items():
            images[k] = (str(v.replace('\\', '/')))
        print(images)
        res = {
            "status": "successs",
            "message": "details get is successsss",
            "Image": images,
            "data":finaldata

        }
        return JsonResponse(res)
    except Exception as e:
        res = {
            "status": "failed",
            "message": str(e)
        }
        return JsonResponse(res)


@api_view(['GET', 'POST', ])
def freezedata(request):
    try:
        request_freeje = json.loads(request.body)
        print("frontend")
        print(request_freeje)
        print("frontend")
        current_date = datetime.datetime.now()
        Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
        Segment_create = models.Segment(
            id=Id_val,
            rule_data=request_freeje,

        )
        Segment_create.save()
        print("after")

        res = {"status": "success", "message": "freeze data successfully", "data": request_freeje}
        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


def get_formated_description(rule_data):
    desription = "if "
    dict_rule_data = ast.literal_eval(rule_data)
    for item in dict_rule_data.get("InputFileds"):
        desription = desription + item.get("selectVariable") + " " + item.get("sampling") + " " + item.get(
            "evaluationValue") + " " + item.get("binaryOperation") + " "
    return desription


@api_view(['GET', 'POST', ])
def sendfreezedata(request):
    try:
        # jobs_res = models.Segment.objects.all().delete()
        operations_res = json.loads(serializers.serialize("json", models.Segment.objects.all()))
        print(operations_res)
        # fields=[]
        field_list = []

        for item in operations_res:
            segment = "segment" + str(len(field_list))
            formated_description = get_formated_description(item.get("fields").get("rule_data"))
            field_list.append({"segment": segment,
                               "modelevaluated": ast.literal_eval(item.get("fields").get("rule_data")).get(
                                   "modelevaluated", " "),
                               "description": formated_description.rstrip(),
                               })

        # for i in operations_res:
        #     l=i["fields"]
        #     m=l["rule_data"]

        #     res = ast.literal_eval(m)
        #     p=res["InputFileds"]
        #     pp=p[0]
        #     mm=res["modelevaluated"]
        #     pp.update([('modelevaluated',mm)])
        #     del pp['id']
        #     print(pp)
        #     fields.append(pp)
        #     res={"InputFileds":fields}
        res = {"status": "success", "message": "freeze data successfully", "InputFileds": field_list}
        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


@api_view(['GET', 'POST', ])
def rulesdata(request):
    try:
        # jobs_res = models.Segment.objects.all().delete()
        # operations_res = json.loads(serializers.serialize("json", models.Rules.objects.all()))
        # print(operations_res)
        # for i in operations_res:
        #    p=i["fields"]
        #    l=p["rule_data"]
        #    res = ast.literal_eval(l)
        #    print(type(res))
        # rem={'Rules':res}

        # res={"status": "success", "message": "freeze data successfully","rules":rem}

        rdata = json.loads(serializers.serialize("json", models.Rules.objects.all()))
        unique_rules = []
        for ritem in ast.literal_eval(rdata[0].get("fields").get("rule_data")):
            if ritem.get("Rule") not in unique_rules:
                unique_rules.append(ritem.get("Rule"))
        rule_data = []
        for urule in unique_rules:
            description = "if "
            for s_item in [item for item in ast.literal_eval(rdata[0].get("fields").get("rule_data")) if
                           item["Rule"] == urule]:
                description = description + s_item.get("selectVariable") + " " + s_item.get(
                    "selectOperator") + " " + s_item.get("evaluationValue") + " " + s_item.get("selectOperation") + " "
            rule_data.append({
                "Rule": urule,
                "selectedVariable": s_item.get("selectedVariable"),
                "description": description.rstrip()
            })

        res = {"status": "success", "message": "freeze data successfully", "rules": rule_data}

        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


@api_view(['GET', 'POST', ])
def download(request):
    try:
        operations_res = json.loads(serializers.serialize("json", models.Segment.objects.all()))
        print(operations_res)
        fields = []
        for i in operations_res:
            segment = "segment" + str(len(fields))
            l = i["fields"]
            m = l["rule_data"]

            res = ast.literal_eval(m)
            p = res["InputFileds"]
            pp = p[0]
            mm = res["modelevaluated"]
            pp.update([('modelevaluated', mm)])
            del pp['id']
            print(pp, "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # fields.append(pp)

            segment_data = ast.literal_eval(m)
            # seg = {"InputFileds": fields}

            for item in segment_data.get("InputFileds"):
                del item["id"]
                item["modelevaluated"] = segment_data.get("modelevaluated")
            del segment_data["modelevaluated"]
            fields.append({segment: segment_data})

            operations_res = json.loads(serializers.serialize("json", models.Rules.objects.all()))
            print(operations_res)
            for i in operations_res:
                p = i["fields"]
                l = p["rule_data"]
                res = ast.literal_eval(l)
                print(type(res))
            rules = {'Rules': res}


        res = { "rules": rules, "segment": fields}

        json_string =json.dumps(res)


        path = "static"

        filepath = os.path.join(path, "Download")
        shutil.rmtree(filepath)
        print("000000000000")
        print(filepath)
        print("0000000000000")
        os.makedirs(filepath, exist_ok=True)
        file="Segment_Rules.json"
        file_path=os.path.join(filepath,file)

        with open(file_path,'w') as fs:
            json.dump(json_string,fs)
        # pythonscript=os.path.join("static","pythonscript")
        # shutil.copytree(pythonscript,filepath)
        modes=os.path.join("static","models_temp")
        # mode=os.path.join(filepath,"models")
        # os.makedirs(mode, exist_ok=True)
        # shutil.copytree(modes, filepath)
        files=os.listdir(modes)
        for name in files:
            shutil.copy2(os.path.join(modes,name),filepath)
        pythonscript=os.path.join("static","pythonscript")
        file = os.listdir(pythonscript)
        for name in file:
            shutil.copy2(os.path.join(pythonscript, name), filepath)
        # scriptfile=os.path.join(filepath,"script")
        # shutil.copytree(pythonscript,scriptfile)
        from shutil import make_archive
        from wsgiref.util import FileWrapper
        file_name="script"
        path_to_zip = make_archive(filepath, "zip", filepath)
        response = HttpResponse(FileWrapper(open(path_to_zip, 'rb')), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="{filename}.zip"'.format(
            filename=file_name.replace(" ", "_")
        )
        print(path_to_zip)
        down={"dowmload_path":"https://edge-detect-v2.herokuapp.com/ static /Download.zip" }
        return response
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


def segmentmodelling(df, seg_models, temp_seg_dict, models):
    df_cat_ = df.select_dtypes(exclude=['float64', 'int64'])
    df_int_ = df.select_dtypes(include=['float64', 'int64'])
    df_cat_.fillna('NA')
    df_int_.fillna(0)
    # One hot encoding for categorical features.
    # df_cat_ = pd.get_dummies(df_cat_)
    df = pd.concat([df_cat_, df_int_], axis=1)
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
        model_name = seg_models['segment' + str(i)]
        # Need to pick up the respective model from models
        filtered_predictions = model.predict(X_filter)
        X_filter['target_variable'] = filtered_predictions
        modeled_df = modeled_df.append(X_filter)

        return modeled_df


@api_view(['GET', 'POST', ])
def rulebased(request):
    try:
        request_json = json.loads(request.body)
        print("++++++++++++++++++", request_json)
        operations_res = json.loads(serializers.serialize("json", models.Sai.objects.all()))
        print("++++++++++++++++++", operations_res)
        print(type(operations_res))
        a = operations_res[0]
        p = a["fields"]
        target_variable = p["target_column"]
        run_models = p["modells"]
        new_list = []
        for i in range(len(request_json['Rules'])):
            Rule = request_json['Rules'][i]['ruleInputs']
            for k in range(len(Rule)):
                Rule[k].update({'Rule': 'Rule' + str(i)})
            new_list.append(Rule)
        print("\\\\\=======///////")
        li = []
        for i in new_list:
            for j in i:
                del j['id']
                print(j)
                li.append(j)
        print(li)
        # rules=[]
        # a=0
        # for i in new_list:
        #     z={a:i}
        #     a = +1
        #     rules.append(z)
        # print(rules)
        jobs_res = models.Rules.objects.all().delete()
        current_date = datetime.datetime.now()
        Id_val = int(current_date.strftime("%Y%m%d%H%M%S"))
        Segment_create = models.Rules(
            id=Id_val,
            rule_data=li,

        )
        Segment_create.save()
        print("after")

        print("////////////////")
        import joblib
        import os
        path = 'static'
        PROJECT_ROOT_DIR = path
        IMAGES_PATHh = os.path.join(PROJECT_ROOT_DIR, "models_temp", "")

        import pickle
        import re

        model_names = os.listdir(IMAGES_PATHh)
        modelss = dict()
        for item in model_names:
            modelss[item[:-4]] = joblib.load(IMAGES_PATHh + item)
        print("PPPPPPPPPPPPPPP", type(modelss))
        print(modelss)

        print("[][]+++++++++++++++++++++++++++[][]")
        print(type(run_models))
        res = ast.literal_eval(run_models)

        res = {"status": "success", "message": "upload created successfully", "rules": "sa", "models": res}
        return JsonResponse(res)
    except Exception as e:
        print("error:", e)
        res = {"status": "failed", "message": str(e)}
        return JsonResponse(res)


def logistic_regression(params={"C": [0.01, 0.1, 1, 10, 100, 1000]},
                        folds=KFold(n_splits=5, shuffle=True, random_state=4)
                        ):
    scaler = StandardScaler()

    train_columns = X_train.columns
    X_train_ = pd.DataFrame(data=scaler.fit_transform(X_train), columns=train_columns)
    X_test_ = pd.DataFrame(data=scaler.fit_transform(X_test), columns=train_columns)
    X_train_ = X_train_.fillna(0)
    X_test_ = X_test_.fillna(0)

    if optimize_hyperparameter == False:
        params = {i: [j] for i, j in LogisticRegression().get_params().items()}

    model_cv = GridSearchCV(estimator=LogisticRegression(),
                            param_grid=params,
                            scoring='roc_auc',
                            cv=folds,
                            n_jobs=-1,
                            verbose=1,
                            return_train_score=True)
    # perform hyperparameter tuning
    model_cv.fit(X_train_, y_train)
    print('Best ROC AUC score: ', model_cv.best_score_)
    # print the optimum value of hyperparameters
    print('Best hyperparameters: ', model_cv.best_params_)

    cv_results = pd.DataFrame(model_cv.cv_results_)
    return (model_cv, cv_results)


def decision_tree(param_grid={'max_depth': range(5, 15, 5), 'min_samples_leaf': range(50, 150, 50),
                              'min_samples_split': range(50, 150, 50), }):
    # Instantiate the grid search model
    #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test

    dtree = DecisionTreeClassifier()

    if optimize_hyperparameter == False:
        param_grid = {i: [j] for i, j in dtree.get_params().items()}

    grid_search = GridSearchCV(estimator=dtree,
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=3,
                               n_jobs=-1,
                               verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    print("Best roc auc score : ", grid_search.best_score_)
    print(grid_search.best_estimator_)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    return (grid_search, cv_results)


def xgboost_classifier(folds=3,
                       param_grid={'learning_rate': [0.2, 0.6],
                                   'subsample': [0.3, 0.6, 0.9]}):
    #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test

    xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

    if optimize_hyperparameter == False:
        param_grid = {i: [j] for i, j in xgb_model.get_params().items()}

    model_cv = GridSearchCV(estimator=xgb_model,
                            param_grid=param_grid,
                            scoring='roc_auc',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)

    model_cv.fit(X_train, y_train)
    print("Best roc auc score : ", model_cv.best_score_)
    print(model_cv.best_estimator_)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    return (model_cv, cv_results)


def gbm_classifier(parameters={
    "loss": ["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse", "mae"],
    "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators": [10]
}, folds=10):
    #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test

    gbm_model = GradientBoostingClassifier()
    print("gbmgbmgmbmmbgm")
    if optimize_hyperparameter == False:
        parameters = {i: [j] for i, j in gbm_model.get_params().items()}
    print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
    model_cv = GridSearchCV(gbm_model, parameters, n_jobs=-1, scoring='roc_auc',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)

    model_cv.fit(X_train, y_train)
    print("Best roc auc score : ", model_cv.best_score_)
    print(model_cv.best_estimator_)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    print(cv_results)
    print("/////////////////////////")
    return (model_cv, cv_results)


def lgbm_classifier(parameters={'learning_rate': [0.01], 'n_estimators': [8, 24],
                                'num_leaves': [6, 8, 12, 16], 'boosting_type': ['gbdt'],
                                'objective': ['binary'], 'seed': [500],
                                'colsample_bytree': [0.65, 0.75, 0.8],
                                'subsample': [0.7, 0.75], 'reg_alpha': [1, 2, 6],
                                'reg_lambda': [1, 2, 6]}, folds=10):
    #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test

    lgb_model = lgb.LGBMClassifier()

    if optimize_hyperparameter == False:
        parameters = {i: [j] for i, j in lgb_model.get_params().items()}

    model_cv = GridSearchCV(lgb_model, parameters, n_jobs=-1, scoring='roc_auc',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)

    model_cv.fit(X_train, y_train)
    print("Best roc auc score : ", model_cv.best_score_)
    print(model_cv.best_estimator_)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    return (model_cv, cv_results)


def cb_classifier(parameters={'iterations': [500],
                              'depth': [4, 5, 6],
                              'loss_function': ['Logloss', 'CrossEntropy'],
                              'l2_leaf_reg': np.logspace(-20, -19, 3),
                              'leaf_estimation_iterations': [10],
                              'logging_level': ['Silent'],
                              'random_seed': [42]
                              }, folds=10):
    #     (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test

    cb_model = CatBoostClassifier()

    if optimize_hyperparameter == False:
        parameters = {i: [j] for i, j in cb_model.get_params().items()}

    model_cv = GridSearchCV(cb_model, parameters, n_jobs=-1, scoring='roc_auc',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)

    model_cv.fit(X_train, y_train)
    print("Best roc auc score : ", model_cv.best_score_)
    print(model_cv.best_estimator_)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    return (model_cv, cv_results)


def rf_classifier(folds=3,
                  param_grid={'bootstrap': [True, False],
                              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                              'max_features': ['auto', 'sqrt'],
                              'min_samples_leaf': [1, 2, 4],
                              'min_samples_split': [2, 5, 10],
                              'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}):
    #         (X_train, X_test, y_train, y_test) = self.X_train, self.X_test, self.y_train, self.y_test

    rf_model = RandomForestClassifier()

    if optimize_hyperparameter == False:
        param_grid = {i: [j] for i, j in rf_model.get_params().items()}

    model_cv = GridSearchCV(estimator=rf_model,
                            param_grid=param_grid,
                            scoring='roc_auc',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)

    model_cv.fit(X_train, y_train)
    print("Best roc auc score : ", model_cv.best_score_)
    print(model_cv.best_estimator_)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    return (model_cv, cv_results)


