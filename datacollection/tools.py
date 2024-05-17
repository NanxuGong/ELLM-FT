import numpy
import numpy as np
import pandas as pd
from collections import defaultdict

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, pairwise_distances
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.special import expit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn
import torch.utils.data as Data
from logger import error, info
from Operation import add_binary, op_map, op_map_r


def cube(x):
    return x ** 3


def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        o = np.reciprocal
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '/':
        o = np.divide
    elif o == '*':
        o = np.multiply
    elif o == 'stand_scaler':
        o = StandardScaler()
    elif o == 'minmax_scaler':
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == 'quan_trans':
        o = QuantileTransformer(random_state=0)
    elif o == 'exp':
        o = np.exp
    elif o == 'cube':
        o = cube
    elif o == 'sigmoid':
        o = expit
    elif o == 'log':
        o = np.log
    else:
        print('Please check your operation!')
    return o


# def feature_distance(features, y):
#     dis_mat = []
#     for i in range(features.shape[1]):
#         tmp = []
#         for j in range(features.shape[1]):
#             tmp.append(np.abs(mutual_info_regression(features[:, i].reshape
#                 (-1, 1), y) - mutual_info_regression(features[:, j].reshape
#                 (-1, 1), y))[0] / (mutual_info_regression(features[:, i].
#                 reshape(-1, 1), features[:, j].reshape(-1, 1))[0] + 1e-05))
#         dis_mat.append(np.array(tmp))
#     dis_mat = np.array(dis_mat)
#     return dis_mat

def eu_distance(feature, y):
    return pairwise_distances(feature.reshape(1, -1), y.reshape(1, -1), metric='euclidean')


def feature_distance(features, y):
    # dis_mat = []
    # for i in range(features.shape[1]):
    #     tmp = []
    #     for j in range(features.shape[1]):
    #         tmp.append(np.abs(eu_distance(features[:, i], features[:, j])))
    #     dis_mat.append(np.array(tmp))
    # dis_mat = np.array(dis_mat)
    r = torch.tensor(features)
    return torch.cdist(r.transpose(-1,0) , r.transpose(-1,0),
                       p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary').numpy()

def cluster_features(features, y, cluster_num=2):
    k = int(np.sqrt(features.shape[1]))
    features = feature_distance(features, y)
    clustering = AgglomerativeClustering(n_clusters=k, affinity=
        'precomputed', linkage='single').fit(features)
    labels = clustering.labels_
    clusters = defaultdict(list)
    for ind, item in enumerate(labels):
        clusters[item].append(ind)
    return clusters

def wocluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        clusters[item].append(ind)
    return clusters

class LinearAutoEncoder(nn.Module):

    def __init__(self, input, hidden, act=torch.relu):
        self.encoder = nn.Linear(input, hidden)
        self.encoder_act = act
        self.decoder = nn.Linear(hidden, input)
        self.decoder_act = act
        super().__init__()

    def forward(self, X):
        return self.decoder_act(self.decoder(self.encoder_act(self.encoder(X)))
            )

    def generate(self, X):
        return self.encoder_act(self.encoder(X))


def Feature_GCN(X):
    """
    group feature 可能有一个cluster内元素为1的情况，这样corr - eye后返回的是一个零矩阵，故在这里设置为0时返回一个1.
    """
    corr_matrix = X.corr().abs()
    if len(corr_matrix) == 1:
        W = corr_matrix
    else:
        corr_matrix[np.isnan(corr_matrix)] = 0
        corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
        sum_vec = corr_matrix_.sum()
        for i in range(len(corr_matrix_)):
            corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
            corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
        W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)
    return Feature


class AutoEncoder(nn.Module):

    def __init__(self, N_feature):
        self.N_feature = N_feature
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(self.N_feature, 16), nn.Tanh
            (), nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(
            8, 16), nn.Tanh(), nn.Linear(16, self.N_feature))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def Feature_AE(X, gpu=-1):
    N_feature = X.shape[1]
    autoencoder = AutoEncoder(N_feature)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    X_tensor = torch.Tensor(X.values)
    if gpu >= 0:
        device = torch.device('cuda:' + str(gpu))
        autoencoder.to(device)
    else:
        device = torch.device('cpu')
    train_loader = Data.DataLoader(dataset=X_tensor, batch_size=128,
        shuffle=True, drop_last=False, num_workers=8)
    for epoch in range(10):
        for x in train_loader:
            b_x = x.view(-1, N_feature).float().to(device)
            encoded, decoded = autoencoder.forward(b_x)
            optimizer.zero_grad()
            loss = loss_func(decoded, b_x)
            loss.backward()
            optimizer.step()
    X_tensor.to(device)
    X_encoded = np.mean(autoencoder.forward(X_tensor)[0].cpu().detach().
        numpy(), axis=1)
    return X_encoded


def feature_state_generation(X, method='mds', gpu=-1):
    if method == 'mds':
        return _feature_state_generation_des(X)
    elif method == 'gcn':
        return Feature_GCN(X)
    elif method == 'ae':
        return Feature_AE(X, gpu)
    elif method == 'mds+ae':
        return numpy.append(Feature_AE(X, gpu),
            _feature_state_generation_des(X))
    elif method == 'mds+ae+gcn':
        state_mds = _feature_state_generation_des(X)
        state_gcn = Feature_GCN(X)
        state_ae = Feature_AE(X, gpu)
        return numpy.append(numpy.append(state_ae, state_gcn), state_mds)
    else:
        error('Wrong feature state method')
        raise Exception('Wrong feature state method')


def _feature_state_generation_des(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
            describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix


def select_meta_cluster1(clusters, X, feature_names, epsilon, dqn_cluster,
    method='mds', gpu=-1):
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
            method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0
            ]
        q_vals.append(q_value)
        cluster_list.append(key)
    if np.random.uniform() > epsilon:
        act_id = np.argmax(q_vals)
    else:
        act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    f_cluster = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    info('current select feature name : ' + str(f_names))
    return action_emb, state_emb, f_cluster, f_names


def select_meta_cluster1_indice(clusters, X, feature_names, epsilon, dqn_cluster,
    method='mds', gpu=-1):
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
            method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0
            ]
        q_vals.append(q_value)
        cluster_list.append(key)
    if np.random.uniform() > epsilon:
        act_id = np.argmax(q_vals)
    else:
        act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    # f_cluster = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    info('current select feature name : ' + str(f_names))
    return action_emb, state_emb, list(clusters[cluster_ind]), f_names

def select_operation(f_cluster, operation_set, dqn_operation, steps_done,
    method='mds', gpu=-1):
    op_state = feature_state_generation(pd.DataFrame(f_cluster), method, gpu)
    op_index = dqn_operation.choose_action(op_state, steps_done)
    op = operation_set[op_index]
    info('current select op : ' + str(op))
    return op_state, op, op_index


def select_meta_cluster2(clusters, X, feature_names, f_cluster1, op_emb,
    epsilon, dqn_cluster, method='mds', gpu=-1):
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster1), method,
        gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
            method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0
            ]
        q_vals.append(q_value)
        cluster_list.append(key)
    act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    f_cluster2 = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    return action_emb, state_emb, f_cluster2, f_names

def select_meta_cluster2_indice(clusters, X, feature_names, f_cluster1, op_emb,
    epsilon, dqn_cluster, method='mds', gpu=-1):
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster1), method,
        gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
            method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value(state_emb, action).detach().numpy()[0
            ]
        q_vals.append(q_value)
        cluster_list.append(key)
    act_id = np.random.randint(0, len(clusters))
    cluster_ind = cluster_list[act_id]
    # f_cluster2 = X[:, list(clusters[cluster_ind])]
    action_emb = action_list[act_id]
    f_names = np.array(feature_names)[list(clusters[cluster_ind])]
    return action_emb, state_emb, list(clusters[cluster_ind]), f_names

def operate_two_features(f_cluster1, f_cluster2, op, op_func, f_names1,
    f_names2):
    if f_cluster1.shape[1] < f_cluster2.shape[1]:
        inds = np.random.randint(0, f_cluster2.shape[1], f_cluster1.shape[1])
        rand_fs = f_cluster2[:, inds]
        rand_names = f_names2[inds]
        f_generate = op_func(f_cluster1, rand_fs)
        final_name = [(str(f1_item) + op + str(rand_names[ind])) for ind,
            f1_item in enumerate(f_names1)]
    elif f_cluster1.shape[1] > f_cluster2.shape[1]:
        inds = np.random.randint(0, f_cluster1.shape[1], f_cluster2.shape[1])
        rand_fs = f_cluster1[:, inds]
        rand_names = f_names1[inds]
        f_generate = op_func(rand_fs, f_cluster2)
        final_name = [(str(f1_item) + op + str(f_names2[ind])) for ind,
            f1_item in enumerate(rand_names)]
    else:
        f_generate = op_func(f_cluster1, f_cluster2)
        final_name = [(str(f1_item) + op + str(f_names2[ind])) for ind,
            f1_item in enumerate(f_names1)]
    return f_generate, final_name


def operate_two_features_new(f_cluster1, f_cluster2, op, op_func, f_names1,
    f_names2):
    feas, feas_names = [], []
    for i in range(f_cluster1.shape[1]):
        for j in range(f_cluster2.shape[1]):
            feas.append(op_func(f_cluster1[:, i], f_cluster2[:, j]))
            feas_names.append(add_binary(op_map_r[op], str(f_names1[i]), str(f_names2[j])))
    feas = np.array(feas)
    feas_names = np.array(feas_names)
    return feas.T, feas_names


# def insert_generated_feature_to_original_feas(feas, f):
#     y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
#     y_label.columns = [feas.columns[len(feas.columns) - 1]]
#     feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
#     final_data = pd.concat([feas, f, y_label], axis=1)
#     return final_data
def insert_generated_feature_to_original_feas(feas, f):
    y_label = feas.iloc[:, -1]
    feas = feas.iloc[:, :-1]
    if not isinstance(f, DataFrame):
        f = pd.DataFrame(f)
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data


def generate_next_state_of_meta_cluster1(X, y, dqn_cluster, cluster_num=2,
    method='mds', gpu=-1):
    clusters = cluster_features(X, y, cluster_num)
    state_emb = feature_state_generation(pd.DataFrame(X), method, gpu)
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
            method, gpu)
        # 这里使用evalnet
        q_value = dqn_cluster.get_q_value_next_state(state_emb, action).detach(
            ).numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
        action_list.append(action)
    #     1. 这里直接argmax 之前是epi+argmax
    act_emb = action_list[np.argmax(q_vals)]
    act_ind = cluster_list[np.argmax(q_vals)]
    f_cluster = X[:, list(clusters[act_ind])]
    return act_emb, state_emb, f_cluster, clusters


def generate_next_state_of_meta_operation(f_cluster_, operation_set,
    dqn_operation, method='mds', gpu=-1):
    op_state = feature_state_generation(pd.DataFrame(f_cluster_), method, gpu)
    op_index = dqn_operation.choose_next_action(op_state)
    op = operation_set[op_index]
    return op_state, op


def generate_next_state_of_meta_cluster2(f_cluster_, op_emb_, clusters, X,
    dqn_cluster, method='mds', gpu=-1):
    feature_emb = feature_state_generation(pd.DataFrame(f_cluster_), method,
        gpu)
    state_emb = torch.cat((torch.tensor(feature_emb), torch.tensor(op_emb_)))
    q_vals, cluster_list, action_list = [], [], []
    for key, value in clusters.items():
        action = feature_state_generation(pd.DataFrame(X[:, list(value)]),
            method, gpu)
        action_list.append(action)
        q_value = dqn_cluster.get_q_value_next_state(state_emb, action).detach(
            ).numpy()[0]
        q_vals.append(q_value)
        cluster_list.append(key)
    action_emb = action_list[np.argmax(q_vals)]
    return action_emb, state_emb


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type, metric_type, state_num=10):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'rank':
        pass
    else:
        return -1


def downstream_task(data, task_type, metric_type, state_num=10):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state=state_num, shuffle=True)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        if metric_type == 'acc':
            return accuracy_score(y_test, y_predict)
        elif metric_type == 'pre':
            return precision_score(y_test, y_predict)
        elif metric_type == 'rec':
            return recall_score(y_test, y_predict)
        elif metric_type == 'f1':
            return f1_score(y_test, y_predict, average='weighted')
    if task_type == 'reg':
        reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        if metric_type == 'mae':
            return mean_absolute_error(y_test, y_predict)
        elif metric_type == 'mse':
            return mean_squared_error(y_test, y_predict)
        elif metric_type == 'rae':
            return 1 - relative_absolute_error(y_test, y_predict)


def downstream_task_cross_validataion(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
        print(scores)
    if task_type == 'reg':
        reg = RandomForestRegressor(random_state=0)
        scores = 1 - cross_val_score(reg, X, y, cv=5, scoring=make_scorer(
            relative_absolute_error))
        print(scores)


def test_task_new(Dg, task='cls', state_num=10):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        acc_list, pre_list, rec_list, f1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            acc_list.append(accuracy_score(y_test, y_predict))
            pre_list.append(precision_score(y_test, y_predict, average=
                'weighted', zero_division=0))
            rec_list.append(recall_score(y_test, y_predict, average='weighted', zero_division=0)
                )
            f1_list.append(f1_score(y_test, y_predict, average='weighted', zero_division=0))
        return np.mean(acc_list), np.mean(pre_list), np.mean(rec_list
            ), np.mean(f1_list)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(mean_absolute_error(y_test, y_predict))
            mse_list.append(mean_squared_error(y_test, y_predict))
            rae_list.append(relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
                ], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='macro'))
            ras.append(roc_auc_score(y_test, y_predict))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras)
    elif task == 'rank':
        pass
    else:
        return -1


def test_task(Dg, task='cls', state_num=10):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state=state_num, shuffle=True)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        acc = accuracy_score(y_test, y_predict)
        pre = precision_score(y_test, y_predict, average='weighted')
        rec = recall_score(y_test, y_predict, average='weighted')
        f1 = f1_score(y_test, y_predict, average='weighted')
        return acc, pre, rec, f1
    elif task == 'reg':
        reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        return mean_absolute_error(y_test, y_predict), mean_squared_error(
            y_test, y_predict, squared=False), relative_absolute_error(y_test,
            y_predict)
    else:
        return -1


def overall_feature_selection(best_features, task_type):
    if task_type == 'reg':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        reg = linear_model.Lasso(alpha=0.1).fit(X, y)
        model = SelectFromModel(reg, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        mae, mse, rae = test_task_new(new_data, task_type)
        info('mae: {:.3f}, mse: {:.3f}, 1-rae: {:.3f}'.format(mae, mse, 1 -
            rae))
    elif task_type == 'cls':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        clf = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        acc, pre, rec, f1 = test_task_new(new_data, task_type)
        info('acc: {:.3f}, pre: {:.3f}, rec: {:.3f}, f1: {:.3f}'.format(acc,
            pre, rec, f1))
    return new_data
