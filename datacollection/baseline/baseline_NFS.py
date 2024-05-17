# single feature transmission
#
# code from aaai
# feature generation => AutoML with LSTM

# 注释掉feature selection部分
import os
import sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('~/jupyter_base/RL-TKDE/')
import torch
from utils.datacollection.logger import *
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)
# info(torch.get_num_threads())
# info(torch.__config__.parallel_info())
import warnings
import math
from nni.utils import merge_parameter
import nni
torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest
from utils.datacollection.DQN import DQN1, DQN2, DDQN1, DDQN2, DuelingDQN1, DuelingDDQN2, DuelingDQN2, DuelingDDQN1
from utils.datacollection.tools import *
from utils.datacollection.task_mapping import task_dict, task_type, base_path, task_measure, state_rep, support_rl_method
import argparse


def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--file-name', type=str, default='housing_boston',
        help='data name')
    parser.add_argument('--log-level', type=str, default='info', help=
        'log level, check the _utils.logger')
    parser.add_argument('--task', type=str, default='ng', help=
        'ng/cls/reg/det/rank, if provided ng, the model will take the task type in config'
        )
    parser.add_argument('--episodes', type=int, default=5, help=
        'episodes for training')
    parser.add_argument('--steps', type=int, default=2, help=
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
    parser.add_argument('--a', type=float, default=0, help='a')
    parser.add_argument('--b', type=float, default=0, help='b')
    parser.add_argument('--c', type=float, default=0, help='c')
    parser.add_argument('--rl-method', type=str, default='dqn', help=
        'used reinforcement methods')
    parser.add_argument('--state-method', type=str, default='mds',
        help='reinforcement state representation method')
    parser.add_argument('--default-cuda', type=int, default=-1, help=
        'the using cuda')
    args, _ = parser.parse_known_args()
    return args


def model_train(param, nni):
    DEVICE = param['default_cuda']
    STATE_METHOD = param['state_method']
    assert STATE_METHOD in state_rep

        # raise
    # start_time = str(time.asctime())
    D_OPT_PATH = './tmp/' + param['file_name'] + '/' + nni.get_trial_id(
        ) + '.csv'
    # info('opt path is {}'.format(D_OPT_PATH))
    OVER_ALL_D_PATH = './tmp/' + param['file_name'] + '/' + nni.get_trial_id(
        ) + '_overall.csv'
    # data_path = base_path + param['file_name'] + '.hdf'
    # info('read the data from {}'.format(data_path))
    data_path = base_path + '/history/' + param['file_name'] + '.hdf'
    # info('read the data from {}'.format(data_path))
    Dg = pd.read_hdf(data_path, 'train')
    Dg_ = pd.read_hdf(data_path, 'test')
    feature_names = [str(i) for i in list(Dg.columns)]
    # info('initialize the features...')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = Dg.values[:, :-1]
    X = scaler.fit_transform(X)
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    
    X_ = Dg_.values[:, :-1]
    X_ = scaler.fit_transform(X_)
    y_ = Dg_.values[:, -1]
    Dg_ = pd.concat([pd.DataFrame(X_), pd.DataFrame(y_)], axis=1)
    
    Dg.columns = feature_names
    O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'stand_scaler',
        'minmax_scaler', 'quan_trans', 'sigmoid', 'log', 'reciprocal']
    O2 = ['+', '-', '*', '/']
    O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']
    operation_set = O1 + O2
    one_hot_op = pd.get_dummies(operation_set)
    operation_emb = defaultdict()
    for item in one_hot_op.columns:
        operation_emb[item] = one_hot_op[item].values
    EPISODES = param['episodes']
    STEPS = param['steps']
    STATE_DIM = 64
    ACTION_DIM = 64
    MEMORY_CAPACITY = param['memory']
    OP_DIM = len(operation_set)
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
    #     'file_name'])
    # info('the chosen reinforcement learning method is ' + param['rl_method'])
    measure = task_measure[task_name]
    # info('the related measurement is ' + measure)
    # info('initialize the model...')
    if STATE_METHOD == 'gcn':
        STATE_DIM = X.shape[0]
        ACTION_DIM = X.shape[0]
    elif STATE_METHOD == 'ae':
        STATE_DIM = X.shape[0]
        ACTION_DIM = X.shape[0]
    elif STATE_METHOD == 'mds+ae':
        STATE_DIM = X.shape[0] + STATE_DIM
        ACTION_DIM = STATE_DIM
    elif STATE_METHOD == 'mds+ae+gcn':
        STATE_DIM = 2 * X.shape[0] + STATE_DIM
        ACTION_DIM = STATE_DIM
    if param['rl_method'] == 'dqn':
        dqn_cluster1 = DQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM,
            MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS,
            MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=
            ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'ddqn':
        dqn_cluster1 = DDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM,
            MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS,
            MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=
            ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'dueling_dqn':
        dqn_cluster1 = DuelingDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=
            ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DuelingDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS,
            MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DuelingDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM
            =ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'dueling_ddqn':
        dqn_cluster1 = DuelingDDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=
            ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DuelingDDQN2(N_STATES=STATE_DIM, N_ACTIONS=
            N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DuelingDDQN1(STATE_DIM=STATE_DIM + OP_DIM,
            ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    old_per = downstream_task_new(Dg, task_name, measure, state_num=22)
    # info('done the base test with performance of {}'.format(old_per))
    episode = 0
    best_per = -1
    D_OPT = Dg
    best_features = []
    D_original = Dg.copy()
    D_original_ = Dg_.copy()
    steps_done = 0
    EPS_START = param['eps_start']
    EPS_END = param['eps_end']
    EPS_DECAY = param['eps_decay']
    CLUSTER_NUM = 4
    duplicate_count = 0
    a, b, c = param['a'], param['b'], param['c']
    # info('initialize the model hyperparameter configure')
    # info(
    #     'epsilon start with {}, end with {}, the decay is {}, the culster num is {}, the duplicate count is {}, the a, b, and c is set to {}, {}, and {}'
    #     .format(EPS_START, EPS_END, EPS_DECAY, CLUSTER_NUM, duplicate_count,
    #     a, b, c))
    # info('the training start...')
    training_start_time = time.time()
    # info('start training at ' + str(training_start_time))
    best_step = -1
    best_episode = -1
    while episode < EPISODES:
        eps_start_time = time.time()
        step = 0
        Dg = D_original.copy()
        Dg_ = D_original_.copy()
        Dg_local = ''
        local_best = -999
        best_per_opt = []
        while step < STEPS:
            step_start_time = time.time()
            steps_done += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 *
                steps_done / EPS_DECAY)
            clusters = wocluster_features(X, y, cluster_num=3)
            print(clusters)
            action_emb_c1, state_emb_c1, f_cluster1_indice, f_names1 = (
                select_meta_cluster1_indice(clusters, Dg.values[:, :-1],
                feature_names, eps_threshold, dqn_cluster1, STATE_METHOD,
                DEVICE))
            f_cluster1 = Dg.values[:, :-1][:, f_cluster1_indice]
            f_cluster1_ = Dg_.values[:, :-1][:, f_cluster1_indice]
            state_emb_op, op, op_index = select_operation(f_cluster1,
                operation_set, dqn_operation, steps_done, STATE_METHOD, DEVICE)
            # info('start operating in step {}'.format(step))
            # info('current op is ' + str(op))
            if op in O1:
                op_sign = justify_operation_type(op)
                f_new, f_new_name, f_new_ = [], [], []
                if op == 'sqrt':
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] < 0) == 0 or np.sum(f_cluster1_[:, i] < 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(f_names1[i] + '_' + op)
                            f_new_.append(op_sign(f_cluster1_[:, i]))
                    f_generate = np.array(f_new).T
                    f_generate_ = np.array(f_new_).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op == 'reciprocal':
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] == 0) == 0 or np.sum(f_cluster1_[:, i] == 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(f_names1[i] + '_' + op)
                            f_new_.append(op_sign(f_cluster1_[:, i]))
                    f_generate = np.array(f_new).T
                    f_generate_ = np.array(f_new_).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op == 'log':
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] <= 0) == 0 or np.sum(f_cluster1_[:, i] <= 0) == 0 :
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(f_names1[i] + '_' + op)
                            f_new_.append(op_sign(f_cluster1_[:, i]))
                    f_generate = np.array(f_new).T
                    f_generate_ = np.array(f_new_).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op in O3:
                    f_generate = op_sign.fit_transform(f_cluster1)
                    f_generate_ = op_sign.fit_transform(f_cluster1_)
                    final_name = [(str(f_n) + '_' + op) for f_n in f_names1]
                else:
                    f_generate = op_sign(f_cluster1)
                    f_generate_ = op_sign(f_cluster1_)
                    final_name = [(str(f_n) + '_' + op) for f_n in f_names1]
            if op in O2:
                op_emb = operation_emb[op]
                op_func = justify_operation_type(op)
                action_emb_c2, state_emb_c2, f_cluster2_indice, f_names2 = (
                    select_meta_cluster2_indice(clusters, Dg.values[:, :-1],
                    feature_names, f_cluster1, op_emb, eps_threshold,
                    dqn_cluster2, STATE_METHOD, DEVICE))
                f_cluster2 = Dg.values[:, :-1][:, f_cluster2_indice]
                f_cluster2_ = Dg_.values[:, :-1][:, f_cluster2_indice]
                if op == '/' and (np.sum(f_cluster2 == 0) > 0 or np.sum(f_cluster2_ == 0) > 0):
                    continue
                f_generate, final_name = operate_two_features_new(f_cluster1,
                    f_cluster2, op, op_func, f_names1, f_names2)
                f_generate_, final_name = operate_two_features_new(f_cluster1_,
                                                                  f_cluster2_, op, op_func, f_names1, f_names2)
            if len(f_generate) > 0 and (np.max(f_generate) > 1000 or np.max(f_generate_) > 1000):
                scaler = MinMaxScaler()
                f_generate = scaler.fit_transform(f_generate)
                f_generate_ = scaler.fit_transform(f_generate_)
            f_generate = pd.DataFrame(f_generate, columns=final_name)
            f_generate_ = pd.DataFrame(f_generate_, columns=final_name)
            public_name = np.intersect1d(np.array(Dg.columns), final_name)
            if len(public_name) > 0:
                reduns = np.setxor1d(final_name, public_name)
                if len(reduns) > 0:
                    f_generate = f_generate[reduns]
                    f_generate_ = f_generate_[reduns]
                    Dg = insert_generated_feature_to_original_feas(Dg,
                        f_generate)
                    Dg_ = insert_generated_feature_to_original_feas(Dg_, f_generate_)
                else:
                    continue
            else:
                Dg = insert_generated_feature_to_original_feas(Dg, f_generate)
                Dg_ = insert_generated_feature_to_original_feas(Dg_, f_generate_)
            feature_names = list(Dg.columns)
            new_per = downstream_task_new(Dg, task_name, measure, state_num=0)
            # nni.report_intermediate_result(new_per)
            reward = new_per - old_per
            # new_per_ = downstream_task_new(Dg_, task_name, measure, state_num=0)
            r_c1, r_op, r_c2 = param['a'] / 10 * reward / 3, param['b'
            ] / 10 * reward / 3, param['c'] / 10 * reward / 3
            if new_per > best_per:
                bset_step = step
                best_episode = episode
                best_per = new_per
                D_OPT = Dg_.copy()
            # if new_per > local_best:
            #     local_best = new_per
            #     Dg_local = Dg.copy()
            old_per = new_per
            action_emb_c1_, state_emb_c1_, f_cluster_, clusters_ = (
                generate_next_state_of_meta_cluster1(Dg.values[:, :-1], y,
                dqn_cluster1, cluster_num=CLUSTER_NUM, method=STATE_METHOD,
                gpu=DEVICE))
            state_emb_op_, op_ = generate_next_state_of_meta_operation(
                f_cluster_, operation_set, dqn_operation, method=
                STATE_METHOD, gpu=DEVICE)
            if op in O2:
                action_emb_c2_, state_emb_c2_ = (
                    generate_next_state_of_meta_cluster2(f_cluster_,
                    operation_emb[op_], clusters_, Dg.values[:, :-1],
                    dqn_cluster2, method=STATE_METHOD, gpu=DEVICE))
                dqn_cluster2.store_transition(state_emb_c2, action_emb_c2,
                    r_c2, state_emb_c2_, action_emb_c2_)
            dqn_cluster1.store_transition(state_emb_c1, action_emb_c1, r_c1,
                state_emb_c1_, action_emb_c1_)
            dqn_operation.store_transition(state_emb_op, op_index, r_op,
                state_emb_op_)
            if dqn_cluster1.memory_counter > dqn_cluster1.MEMORY_CAPACITY:
                dqn_cluster1.learn()
            if dqn_cluster2.memory_counter > dqn_cluster2.MEMORY_CAPACITY:
                dqn_cluster2.learn()
            if dqn_operation.memory_counter > dqn_operation.MEMORY_CAPACITY:
                dqn_operation.learn()
            # if Dg.shape[1] > FEATURE_LIMIT:
#             #     selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT
            #         ).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
            #     cols = selector.get_support()
            #     X_new = Dg.iloc[:, :-1].loc[:, cols]
            #     Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
            # info(
            #     'New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}'
            #     .format(new_per, best_per, best_episode, best_step, base_per))
            # info('Episode {}, Step {} ends!'.format(episode, step))
            best_per_opt.append(best_per)
            # info('Current spend time for step-{} is: {:.1f}s'.format(step,
            #     time.time() - step_start_time))
            step += 1
        # if episode != EPISODES - 1:
        #     best_features.append(pd.DataFrame(Dg_local.iloc[:, :-1]))
        # else:
        #     best_features.append(Dg_local)
        episode += 1
        # info('Current spend time for episode-{} is: {:.1f}s'.format(episode,
        #     time.time() - eps_start_time))
        # if episode % 5 == 0:
            # info('Best performance is: {:.6f}'.format(np.min(best_per_opt)))
            # info('Episode {} ends!'.format(episode))
        episode += 1
    # info('Total spend time for is: {:.1f}s'.format(time.time() -
    #     training_start_time))
    # info('Exploration ends!')
    info('Begin evaluation...')
    info(param['file_name'])
    if task_name == 'reg':
        mae0, rmse0, rae0 = test_task_new(Dg_, task=task_name,
            state_num=0)
        mae1, rmse1, rae1 = test_task_new(D_OPT, task=task_name, state_num=0)
        # nni.report_final_result(1 - rae1)
        info('MAE on original is: {:.3f}, MAE on generated is: {:.3f}'.
            format(mae0, mae1))
        info('RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}'.
            format(rmse0, rmse1))
        info('1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}'.
            format(1 - rae0, 1 - rae1))
    elif task_name == 'cls':
        acc0, precision0, recall0, f1_0 = test_task_new(Dg_, task=
            task_name, state_num=0)
        acc1, precision1, recall1, f1_1 = test_task_new(D_OPT, task=
            task_name, state_num=0)
        # nni.report_final_result(f1_1)
        info('Acc on original is: {:.3f}, Acc on generated is: {:.3f}'.
            format(acc0, acc1))
        info('Pre on original is: {:.3f}, Pre on generated is: {:.3f}'.
            format(precision0, precision1))
        info('Rec on original is: {:.3f}, Rec on generated is: {:.3f}'.
            format(recall0, recall1))
        info('F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}'.
            format(f1_0, f1_1))
    elif task_name == 'det':
        map0, f1_0, ras0 = test_task_new(Dg_, task=task_name,
            state_num=0)
        map1, f1_1, ras1 = test_task_new(D_OPT, task=task_name, state_num=0)
        # nni.report_final_result(ras1)
        info(
            'Average Precision Score on original is: {:.3f}, Average Precision Score on generated is: {:.3f}'
            .format(map0, map1))
        info(
            'F1 Score on original is: {:.3f}, F1 Score on generated is: {:.3f}'
            .format(f1_0, f1_1))
        info(
            'ROC AUC Score on original is: {:.3f}, ROC AUC Score on generated is: {:.3f}'
            .format(ras0, ras1))
    else:
        error('wrong task name!!!!!')
        assert False
    # info('Total using time: {:.1f}s'.format(time.time() - training_start_time))
    # D_OPT.to_csv(D_OPT_PATH)


if __name__ == '__main__':
    for name in [
        # 'airfoil', 'amazon_employee',
        #                                                            'ap_omentum_ovary', 'german_credit',
        #                                                            'higgs',
        #          'housing_boston',
        # 'ionosphere',
        #                                                            'lymphography',
        # 'messidor_features', 'openml_620',
        #                                                            'pima_indian', 'spam_base', 'spectf', 'svmguide3',
        #                                                            'uci_credit_card',
        'wine_red', 'wine_white',
                                                                   'openml_586',
                                                                   'openml_589', 'openml_607', 'openml_616',
                                                                   'openml_618',
                                                                   'openml_637']:
        try:
            args = init_param()
            tuner_params = nni.get_next_parameter()
            trail_id = nni.get_trial_id()
            params = vars(merge_parameter(args, tuner_params))
            params['file_name'] = name
            start_time = str(time.asctime())
            debug(tuner_params)
            # info(params)
            model_train(params, nni)
        except Exception as exception:
            error(exception)
            raise
