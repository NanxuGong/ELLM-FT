from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation
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
fh = logging.FileHandler('lda_log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
files = {
    # 'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':
    # 'cls', 'german_credit': 'cls', 'higgs': 'cls',
    #           'ionosphere': 'cls', 'lymphography': 'cls',
    #          'messidor_features': 'cls', 'pima_indian': 'cls',
    #          'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
    #          'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
'housing_boston': 'reg',
         'openml_620': 'reg',
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_ = train_data.iloc[:, :-1].values
    X_train_ = scaler.fit_transform(X_train_)
    y_train_ = train_data.iloc[:, -1].values
    X = test_data.iloc[:, :-1].values
    X = scaler.fit_transform(X)
    y = test_data.iloc[:, -1].values
    if task_type == 'cls':
        lda = LatentDirichletAllocation(n_components=5, random_state=0)
        lda.fit(X_train_)
        X_new = lda.transform(X)
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
        lda = LatentDirichletAllocation(n_components=5, random_state=0)
        lda.fit(X_train_)
        X_new = lda.transform(X)
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0, max_depth=7)
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
    # if task_type == 'det':
    #     lda = LatentDirichletAllocation(n_components=15, random_state=0)
    #     X_new = lda.fit_transform(X, y.flatten())
    #     kf = KFold(n_splits=5, random_state=0, shuffle=True)
    #     knn_model = KNeighborsClassifier(n_neighbors=5)
    #     ras = []
    #     for train, test in kf.split(X_new):
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
    logger.info('------------------------------------------------')
    print('------------------------------------------------')
