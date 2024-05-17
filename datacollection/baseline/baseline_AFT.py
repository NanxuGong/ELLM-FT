from autofeat import AutoFeatRegressor, AutoFeatModel
from autofeat import AutoFeatClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os
import time
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt=
    '%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler('autofeat_log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
files = {
    # 'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':
    # 'cls', 'german_credit': 'cls', 'higgs': 'cls',
    #          'housing_boston': 'reg',
    #      'ionosphere': 'cls',
    # 'lymphography': 'cls',
    #          'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
    #          'spam_base': 'cls',
    # 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
             }


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


for name, task_type in files.items():
    test_data = pd.read_hdf('./data/history/' + name + '.hdf', key='test')
    train_data = pd.read_hdf('./data/history/' + name + '.hdf', key='train')
    if name == 'ionosphere':
        train_data.drop(columns=1,inplace=True)
        test_data.drop(columns=1,inplace=True)
    # test_data.fillna(test_data.mean(), inplace=True)
    # train_data.fillna(train_data.mean(), inplace=True)
    # del data['casual']
    # del data['registered']
    scaler = StandardScaler()
    X_train = train_data.iloc[:, :-1].values
    X_train = scaler.fit_transform(X_train)
    y_train = train_data.iloc[:, -1].values
    X = test_data.iloc[:, :-1].values
    X = scaler.fit_transform(X)
    y = test_data.iloc[:, -1].values
    if max(y) > 1:
        is_mc = True
    else:
        is_mc = False
    # transformations = ('1/', "exp", "log", "abs", "sqrt", "^2", "^3")
    if task_type == 'cls':
        if is_mc:
            model = AutoFeatModel(n_jobs=8, verbose=1, feateng_steps=1,
            featsel_runs=2, problem_type='mc')
        else:
            model = AutoFeatClassifier(n_jobs=8, verbose=1, feateng_steps=1,
                                       featsel_runs=2)
        model.fit(X_train, y_train.flatten())
        X_new = model.transform(X).values
        
        clf = RandomForestClassifier(random_state=0)
        result = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X_new, y):
            X_train, y_train, X_test, y_test = X_new[train, :], y[train
                ], X_new[test, :], y[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            result.append(f1_score(y_test, y_predict, average='weighted'))
        logger.info(name + ':' + task_type)
        logger.info('5 fold f1:' + str(result))
        logger.info('Final F1:' + str(np.mean(result)))
        print(name + ':' + task_type)
        print('5 fold f1:' + str(result))
        print('Final F1:' + str(np.mean(result)))
    if task_type == 'reg':
        model = AutoFeatRegressor(n_jobs=8, verbose=1, feateng_steps=1,
            featsel_runs=2)
        model.fit(X_train, y_train.flatten())
        X_new = model.transform(X).values
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X_new):
            X_train, y_train, X_test, y_test = X_new[train, :], y[train
                ], X_new[test, :], y[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        logger.info(name + ':' + task_type)
        logger.info('5 fold rae:' + str(rae_list))
        logger.info('Final rae:' + str(np.mean(rae_list)))
        print(name + ':' + task_type)
        print('5 fold rae:', rae_list)
        print('Final rae:', np.mean(rae_list))
    logger.info('------------------------------------------------')
    print('------------------------------------------------')
    # if task_type == 'det':
    #     model = AutoFeatRegressor(n_jobs=8, verbose=1, feateng_steps=1,
    #                               featsel_runs=2)
    #     X_new = model.fit_transform(X, y.flatten()).values
    #     kf = KFold(n_splits=5, random_state=0, shuffle=True)
    #     knn_model = KNeighborsClassifier(n_neighbors=5)
    #     ras = []
    #     for train, test in kf.split(X_new):
    #         print(train)
    #         print(test)
    #         X_train, y_train, X_test, y_test = X_new[train, :], y[train
    #             ], X_new[test, :], y[test]
    #         knn_model.fit(X_train, y_train)
    #         y_predict = knn_model.predict(X_test)
    #         ras.append(roc_auc_score(y_test, y_predict))
    #     logger.info(name + ':' + task_type)
    #     logger.info('5 fold average_precision_score:' + str(ras))
    #     logger.info('Final average_precision_score:' + str(np.mean(ras)))
    #     print(name + ':' + task_type)
    #     print('5 fold average_precision_score:', str(ras))
    #     print('Final average_precision_score:', str(np.mean(ras)))