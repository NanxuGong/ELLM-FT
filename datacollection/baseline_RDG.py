# 随机初始化 、 去掉m-cluster 、 DQN 随机选择
# 随机选择了一个操作，然后把所有feature做了转换，再用feature selection选择操作
import os
import sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('~/jupyter_base/RL-TKDE/')
import torch
from logger import *
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)
# info(torch.get_num_threads())
# info(torch.__config__.parallel_info())
import warnings
import nni
torch.manual_seed(0)
warnings.filterwarnings('ignore')
# warnings.warn('DelftStack')
# warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest
from tools import *
from task_mapping import task_dict, task_type, base_path, task_measure, state_rep, support_rl_method
import argparse


def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--file-name', type=str, default='openml_616',
        help='data name')
    parser.add_argument('--log-level', type=str, default='info', help=
        'log level, check the _utils.logger')
    parser.add_argument('--task', type=str, default='ng', help=
        'ng/cls/reg/det/rank, if provided ng, the model will take the task type in config'
        )
    parser.add_argument('--episodes', type=int, default=3, help=
        'episodes for training')
    parser.add_argument('--steps', type=int, default=0, help=
        'steps for each episode')
    parser.add_argument('--enlarge_num', type=int, default=4, help=
        'feature space enlarge')
    parser.add_argument('--memory', type=int, default=8, help='memory capacity'
        )
    parser.add_argument('--eps_start', type=float, default=0.9, help=
        'eps start')
    parser.add_argument('--eps_end', type=float, default=0.5, help='eps end')
    parser.add_argument('--eps_decay', type=int, default=100, help='eps decay')
    parser.add_argument('--index', type=float, default=0.5, help='file index')
    parser.add_argument('--state', type=int, default=0, help='random_state')
    parser.add_argument('--cluster_num', type=int, default=0, help=
        'cluster_num')
    parser.add_argument('--a', type=float, default=1, help='a')
    parser.add_argument('--b', type=float, default=1, help='b')
    parser.add_argument('--c', type=float, default=1, help='c')
    parser.add_argument('--rl-method', type=str, default='dqn', help=
        'used reinforcement methods')
    parser.add_argument('--state-method', type=str, default='mds',
        help='reinforcement state representation method')
    parser.add_argument('--default-cuda', type=int, default=-1, help=
        'the using cuda')
    args, _ = parser.parse_known_args()
    return args


def model_train(param, id):
    always_best = []
    # DEVICE = param['default_cuda']
    STATE_METHOD = param['state_method']
    assert STATE_METHOD in state_rep
    base_tmp_path = './tmp/'  + params['file_name']
        # raise
    # start_time = str(time.asctime())
    D_OPT_PATH = base_tmp_path 
    # info('opt path is {}'.format(D_OPT_PATH))

    data_path = base_path + param['file_name'] + '.hdf'
    # info('read the data from {}'.format(data_path))
    Dg = pd.read_hdf(data_path)
    test_Dg = pd.read_hdf(data_path)
    feature_names = list(Dg.columns)
    
    # info('initialize the features...')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = Dg.values[:, :-1]
    X = scaler.fit_transform(X)
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    X_ = test_Dg.values[:, :-1]
    X_ = scaler.fit_transform(X_)
    y_ = test_Dg.values[:, -1]
    test_Dg = pd.concat([pd.DataFrame(X_), pd.DataFrame(y_)], axis=1)

    Dg.columns = feature_names
    Dg.columns = Dg.columns.astype(str)
    feature_names = list(Dg.columns)  # Update feature_names
    test_Dg.columns = feature_names  # Use updated feature_names
    O1 = ['sqrt', 'sin', 'cos', 'tanh', 'stand_scaler',
        'minmax_scaler', 'sigmoid', 'log', 'reciprocal', 'none']
    O2 = ['+', '-', '*', '/']
    O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']
    operation_set = O1 + O2
    one_hot_op = pd.get_dummies(operation_set)
    operation_emb = defaultdict()
    for item in one_hot_op.columns:
        operation_emb[item] = one_hot_op[item].values
    EPISODES = param['episodes']
    STEPS = param['steps']
    FEATURE_LIMIT = Dg.shape[1] * param['enlarge_num']
    N_ACTIONS = len(operation_set)
    dqn_cluster1 = None
    dqn_operation = None
    dqn_cluster2 = None
    assert param['rl_method'] in support_rl_method
    if param['task'] == 'ng':
        task_name = task_dict[param['file_name']]
    else:
        assert param['task'] in task_type
        task_name = param['task']
    # info('the task is performing ' + task_name + ' on _dataset ' + param[
        # 'file_name'])
    # info('the chosen reinforcement learning method is ' + param['rl_method'])
    measure = task_measure[task_name]
    # info('the related measurement is ' + measure)
    old_per = downstream_task_new(Dg, task_name, measure, state_num=22)
    # info('done the base test with performance of {}'.format(old_per))
    episode = 0
    step = 0
    best_per = -1
    D_OPT = Dg
    best_features = []
    D_original = Dg.copy()
    D_original_ = test_Dg.copy()
    steps_done = 0
    training_start_time = time.time()
    while episode < EPISODES:
        step = 0
        Dg = D_original.copy()
        Dg_ = D_original_.copy()
        f_generate = Dg.values[:, :-1]
        f_generate_ = Dg_.values[:, :-1]
        # print('ori', f_generate.shape)
        op = operation_set[np.random.randint(0, N_ACTIONS)]
        steps_done += 1
        if op == 'none':
            step += 1
            continue
        if op in O1:
            op_sign = justify_operation_type(op)
            f_new = []
            f_new_ = []
            if op == 'sqrt':
                for i in range(f_generate.shape[1]):
                    if np.sum(f_generate[:, i] < 0) == 0 and np.sum(f_generate_[:, i] < 0) == 0:
                        f_new.append(op_sign(f_generate[:, i]))
                        f_new_.append(op_sign(f_generate_[:, i]))
                        # f_new_name.append(final_name[i] + '_' + op)
                f_generate = np.array(f_new).T
                f_generate_ = np.array(f_new_).T
                # final_name = f_new_name
                if len(f_generate) == 0:
                    continue
            elif op == 'reciprocal':
                for i in range(f_generate.shape[1]):
                    if np.sum(f_generate[:, i] == 0) == 0 and np.sum(f_generate_[:, i] < 0) == 0:
                        f_new.append(op_sign(f_generate[:, i]))
                        f_new_.append(op_sign(f_generate_[:, i]))
                        # f_new_name.append(final_name[i] + '_' + op)
                f_generate = np.array(f_new).T
                f_generate_ = np.array(f_new_).T
                # final_name = f_new_name
                if len(f_generate) == 0:
                    continue
            elif op == 'log':
                for i in range(f_generate.shape[1]):
                    if np.sum(f_generate[:, i] <= 0) == 0 and np.sum(f_generate_[:, i] <= 0) == 0:
                        f_new.append(op_sign(f_generate[:, i]))
                        f_new_.append(op_sign(f_generate_[:, i]))
                        # f_new_name.append(final_name[i] + '_' + op)
                f_generate = np.array(f_new).T
                f_generate_ = np.array(f_new_).T
                # final_name = f_new_name
                if len(f_generate) == 0:
                    continue
            elif op in O3:
                f_generate = op_sign.fit_transform(f_generate)
                f_generate_ = op_sign.fit_transform(f_generate_)
                # final_name = [(str(f_n) + '_' + str(op)) for f_n in final_name]
            else:
                # print('before op', op , f_generate.shape)
                f_generate = op_sign(f_generate)
                f_generate_ = op_sign(f_generate_)
                # print('after op', op , f_generate.shape)
                # final_name = [(str(f_n) + '_' + op) for f_n in final_name]
        if op in O2:
            # continue
            op_func = justify_operation_type(op)
            if op == '/' and np.sum(f_generate == 0) > 0 and np.sum(f_generate_ == 0) > 0:
                continue
            feas = []
            feas_ = []
            for i in range(f_generate.shape[1]):
                for j in range(f_generate.shape[1]):
                    feas.append(op_func(f_generate[:, i], f_generate[:, j]))
            for i in range(f_generate_.shape[1]):
                for j in range(f_generate_.shape[1]):
                    feas_.append(op_func(f_generate_[:, i], f_generate_[:, j]))
            f_generate = np.array(feas).T
            f_generate_ = np.array(feas_).T
        # print('before min max feature', f_generate.shape)
        if len(f_generate) > 0 and (np.max(f_generate) > 1000 or np.max(f_generate_) > 1000):
            scaler = MinMaxScaler()
            f_generate = scaler.fit_transform(f_generate)
            f_generate_ = scaler.fit_transform(f_generate_)
        f_generate = pd.DataFrame(f_generate)
        f_generate_ = pd.DataFrame(f_generate_)
        Dg = insert_generated_feature_to_original_feas(Dg, f_generate)
        Dg_ = insert_generated_feature_to_original_feas(Dg_, f_generate_)
        # if Dg.shape[1] > FEATURE_LIMIT:
        #     selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT
        #         ).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
        #     cols = selector.get_support()
        #     X_new = Dg.iloc[:, :-1].loc[:, cols]
        #     Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
        Dg.columns = Dg.columns.astype(str)
        new_per = downstream_task_new(Dg, task_name, measure, state_num=0)
        always_best.append((Dg.columns, new_per,episode,0))
        if new_per > best_per:
            best_per = new_per
            D_OPT = Dg_.copy()
        print(episode)
        episode += 1
    sep_token = 4
    current_trial_name = 'STANDALONE'
    with open(D_OPT_PATH + '/' + f'{current_trial_name}.bdata', 'w') as f:
        for col_name, per, epi, step_ in always_best:
            col_name = [str(i) for i in list(col_name)]
            line = str.join(f',{str(sep_token)},', col_name) + f',{per},{epi},{step_}\n'
            f.write(line)
    # if task_name == 'reg':
    #     mae0, rmse0, rae0 = test_task_new(test_Dg, task=task_name,
    #         state_num=0)
    #     mae1, rmse1, rae1 = test_task_new(D_OPT, task=task_name, state_num=0)
    #     # nni.report_final_result(1 - rae1)
    #     info('MAE on original is: {:.3f}, MAE on generated is: {:.3f}'.
    #         format(mae0, mae1))
    #     info('RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}'.
    #         format(rmse0, rmse1))
    #     info('1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}'.
    #         format(1 - rae0, 1 - rae1))
    # elif task_name == 'cls':
    #     acc0, precision0, recall0, f1_0 = test_task_new(test_Dg, task=
    #         task_name, state_num=0)
    #     acc1, precision1, recall1, f1_1 = test_task_new(D_OPT, task=
    #         task_name, state_num=0)
    #     # nni.report_final_result(f1_1)
    #     info('Acc on original is: {:.3f}, Acc on generated is: {:.3f}'.
    #         format(acc0, acc1))
    #     info('Pre on original is: {:.3f}, Pre on generated is: {:.3f}'.
    #         format(precision0, precision1))
    #     info('Rec on original is: {:.3f}, Rec on generated is: {:.3f}'.
    #         format(recall0, recall1))
    #     info('F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}'.
    #         format(f1_0, f1_1))
    # # elif task_name == 'det':
    # #     map0, f1_0, ras0 = test_task_new(D_original, task=task_name,
    # #         state_num=0)
    # #     map1, f1_1, ras1 = test_task_new(D_OPT, task=task_name, state_num=0)
    # #     # nni.report_final_result(ras1)
    # #     info(
    # #         'Average Precision Score on original is: {:.3f}, Average Precision Score on generated is: {:.3f}'
    # #         .format(map0, map1))
    # #     info(
    # #         'F1 Score on original is: {:.3f}, F1 Score on generated is: {:.3f}'
    # #         .format(f1_0, f1_1))
    # #     info(
    # #         'ROC AUC Score on original is: {:.3f}, ROC AUC Score on generated is: {:.3f}'
    # #         .format(ras0, ras1))
    # # else:
    # #     error('wrong task name!!!!!')
    # #     assert False
    # info('Total using time: {:.1f}s'.format(time.time() - training_start_time))
    # D_OPT.to_csv(D_OPT_PATH)


if __name__ == '__main__':
    args = init_param()
    # tuner_params = nni.get_next_parameter()
    trail_id = nni.get_trial_id()
    params = vars(args)
    # if not os.path.exists('./log/'):
    #     os.mkdir('./log/')
    # if not os.path.exists('./tmp'):
    #     os.mkdir('./tmp/')
    name = 'openml_616'
    params['file_name'] = name
    params['baseline_name'] = 'erg'
    start_time = str(time.asctime())
    # if not os.path.exists('./log/' + params['file_name'] + '_baseline'):
    #     os.mkdir('./log/' + params['file_name'] + '_baseline')
    # log_file = './log/' + params['file_name'] + '_baseline/' + start_time + '_' + params['baseline_name'] + '.log'
    # logging.basicConfig(filename=log_file, level=logging_level[params[
    #     'log_level']], format=
    #     '%(asctime)s - %(levelname)s : %(message)s', datefmt=
    #     '%Y/%m/%d %H:%M:%S')
    # logger = logging.getLogger('')
    # if not os.path.exists('./tmp/'  + params['file_name'] + '_baseline'):
    #     os.mkdir('./tmp/'  + params['file_name'] + '_baseline')
    # debug(tuner_params)
    info(params)
    model_train(params, trail_id)
