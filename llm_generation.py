
from transformers import AutoTokenizer
import transformers
from Operation import *
from island import *
import torch
import re
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import argparse
import logging
# 配置日志模块
logging.basicConfig(filename='svmguide.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser(description='PyTorch Experiment')
parser.add_argument('--task_name', type=str, default='openml_616',
                    help='data name')
parser.add_argument('--task_type', type=str, default='reg')
parser.add_argument('--generation_num', type=int, default=30)
parser.add_argument('--ind_num', type=int, default=15)
parser.add_argument('--remove_time', type=int, default=2)
parser.add_argument('--update_time', type=int, default=5)
parser.add_argument('--temperature', type=float, default=0.8)
args = parser.parse_args()


model = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error

def downstream_task_new(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0, n_jobs=128)
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
        reg = RandomForestRegressor(random_state=0, n_jobs=128)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    elif task_type == 'rank':
        pass
    else:
        return -1

def create_dialogue_prompt(system_prompt, current_use_msg):
    # first_pair = user_model_qa[0]
    dialogue = f"<s>[INST] <<SYS>> " \
               f"{system_prompt} " \
               f"<</SYS>> " \
            #    f"{first_pair[0]} [/INST] {first_pair[1]} </s>"
    dialogue += "\n"

    dialogue += "<s>[INST] "
    dialogue += current_use_msg
    dialogue += " [/INST]"
    return dialogue


def do_dialogu(system_msg, current_use_msg):
    test_prompt = create_dialogue_prompt(system_msg, current_use_msg)
    sequences = pipeline(
        f'{test_prompt}\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )

    prompt_length = len(test_prompt)
    generated_part = sequences[0]['generated_text'][prompt_length:]
    generated_part = generated_part.strip()
    return generated_part

def main():
    collect_data(args.task_name)
    prompts = []
    accs = []
    df = pandas.read_hdf('data/' + args.task_name + '.hdf')
    y = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    
    with open('prompt.txt', 'r') as f:
        for line in f:
            prompts.append(line + ' \n')

    with open('acc.txt', 'r') as f:
        for line in f:
            accs.append(float(line))
    max_acc = 0
    baseline_acc = max(accs)
    num_ge = 0
    island_g = island_group()
    new_island = island()
    for i in range(len(prompts)):
        new_island.add_prompt(prompt(prompts[i], accs[i]))
        if i % args.ind_num == args.ind_num - 1:
            island_g.add_island(new_island)
            new_island = island()
            new_island.add_prompt(prompt(prompts[i], accs[i]))
    print('Construct', len(island_g.islands), 'islands')

    system = 'you can transfer features to get a new feature set which is represented by postfix expression. Here are features (f0,f1...,fn) and opearations (sqrt, square, sin, cos, tanh, stand_scaler, minmax_scaler, quan_trans, sigmoid, log, reciprocal, cube, +, -, *, /). Everytime I will give you two new feature set examples (the latter, the better), please give me one better according them.'

    acc_list = []
    num_change = 0
    for i in range(args.generation_num):
        now_num = len(prompts)
        print('-------------------------------- generation', i, '--------------------------------')
        for isl in island_g.islands:
            prt = isl.get_prompts()
            output = do_dialogu(system, prt)
            i += 1
            new_trans = re.findall(r'\[(.*?)\]', output)
            if len(new_trans) == 0:
                print('no transformation')
            else:
                for trans in new_trans:
                    new_text = '[' + trans + ']'
                    
                    if isl.is_repeat(new_text):
                        print('no new transformation')
                    else:
                        print(trans)
                        trans = trans.replace('f', '')
                        trans = trans.split(',')
                        is_valid = True
                        new_data = pandas.DataFrame()
                        for tran in trans:
                            # print(tran)
                            try:
                                ops = show_ops_r(converge(tran.split()))
                                if not check_valid(ops):
                                    print('no valid transforamtion') 
                                    is_valid = False
                                    break
                                else:
                                    new_data[tran] = op_post_seq(df,ops)
                            except Exception as e:
                                print('no valid transformation') 
                                is_valid = False
                                break
                        if is_valid:
                            try:
                                new_data['target'] = y
                                acc = downstream_task_new(new_data, args.task_type)
                                if acc>max_acc:
                                    num_change += 1
                                    max_acc = acc
                                    print('--------------------------------')
                                    print('higher accuracy is found!')
                                    print('--------------------------------')
                                new_prt = prompt(new_text, acc)
                                prompts.append(new_text)
                                accs.append(acc)
                                acc_list.append(acc)
                                isl.add_prompt(new_prt)
                                num_ge += 1
                                print('new feature transformation is found and accuracy is ', acc, 'the max accuracy has been changed',  num_change, 'times the max accuracy is', max_acc, 'the baseline accuracy is', baseline_acc, 'the number of generation is', num_ge)
                            except Exception as e:
                                print('no valid transformation') 
        if i % args.remove_time == args.remove_time - 1 and i != 0:
            for isl in island_g.islands:
                isl.remove() 
        if i % args.update_time == args.update_time - 1 and i != 0:
            island_g.island_update()
        print(len(prompts)-now_num,'feature transformation is found in this generation')               
                
    print(len(prompts))
    logging.info("number: %s, accuracy: %s", len(prompts)-450, max_acc)
    # plt.plot(acc_list)
    # plt.title('Accuracy List')
    # plt.xlabel('Index')
    # plt.ylabel('Accuracy')
    # plt.show()
if __name__ == '__main__':
    main()
































