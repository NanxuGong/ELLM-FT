import numpy as np
import pandas
from scipy.special import expit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import sys
import random
O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'stand_scaler',
      'minmax_scaler', 'quan_trans', 'sigmoid', 'log', 'reciprocal', 'cube']
O2 = ['+', '-', '*', '/']
O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']


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


sos_token = 0
eos_token = 1
l_sep_token = 2
r_sep_token = 3
sep_token = 4

operation_set = O1 + O2
op_map = dict()
op_map_r = dict()

for j, i in enumerate(operation_set):
    op_map[j + 5] = i
    op_map_r[i] = j + 5


def add_unary(op_index, pos_1_str):
    if isinstance(pos_1_str, int):
        pos_1_str = str(pos_1_str + len(operation_set) + 5)
    return f'{l_sep_token},{pos_1_str},{op_index},{r_sep_token}'


def add_binary(op_index, pos_1_str, pos_2_str):
    if isinstance(pos_1_str, int):
        pos_1_str = str(pos_1_str + len(operation_set) + 5)
    if isinstance(pos_2_str, int):
        pos_2_str = str(pos_2_str + len(operation_set) + 5)
    return f'{l_sep_token},{pos_1_str},{op_index},{pos_2_str},{r_sep_token}'


def add_group_unary(op_index, g1):
    [add_unary(op_index, i) for i in g1]


def add_group_binary(op_index, g_1, g_2):
    ret = []
    for pos_1 in g_1:
        for pos_2 in g_2:
            ret.append(add_binary(op_index, pos_1, pos_2))


def show_ops(process_seq):
    op_str = []
    for process in process_seq:
        if process == l_sep_token:
            op_str.append('(')
        elif process == r_sep_token:
            op_str.append(')')
        elif process > 3 and process < len(operation_set) + 5:
            op_str.append(op_map[process])
        else:
            op_str.append(str(process - len(operation_set) - 5))
    return op_str


def show_ops_r(process_seq):
    op_str = []
    for process in process_seq:
        if process == '(':
            op_str.append(l_sep_token)
        elif process == ')':
            op_str.append(r_sep_token)
        elif process in operation_set:
            op_str.append(op_map_r[process])
        else:
            op_str.append(int(process) + len(operation_set) + 5)
    return op_str


def op_seq(df, process_seq):
    num_stack = []
    op_stack = []
    for process in process_seq:
        if process >= (len(operation_set) + 5):  # number
            num_stack.append(df.iloc[:, process - len(operation_set) - 5])
        elif process == l_sep_token:
            continue
        elif process == r_sep_token:
            op_name = op_map[op_stack.pop(-1)]
            op_sign = justify_operation_type(op_name)
            if op_name in O1:  # UNARY
                pos_1 = num_stack.pop(-1)
                num_stack.append(op_sign(pos_1))
            else:
                pos_1 = num_stack.pop(0)
                pos_2 = num_stack.pop(0)
                num_stack.append(op_sign(pos_1, pos_2))
        else:
            op_stack.append(process)
    return num_stack.pop(-1)


def check_post_valid(op):
    pass


def _operate_two_features(f1, f2, op_func, op):
    if op == '/' and np.sum(f2 == 0) > 0:
        return None
    return op_func(f1, f2)


def _operate_one_feature(f1, op_func, op):
    if op == 'sqrt':
        if np.sum(f1 < 0) == 0:
            return op_func(f1)
    elif op == 'reciprocal':
        if np.sum(f1 == 0) == 0:
            return op_func(f1)
    elif op == 'log':
        if np.sum(f1 <= 0) == 0:
            return op_func(f1)
    elif op in O3:
        return pandas.DataFrame(op_func.fit_transform(f1.values.reshape(-1,1)).reshape(-1))[0]
    else:
        return op_func(f1)
    return None


def op_post_seq(df, process_seq):
    s = []
    for i in process_seq:
        if i >= (len(operation_set) + 5):  # number
            s.append(df.iloc[:, i - len(operation_set) - 5])
        else:
            op_name = op_map[i]
            op_sign = justify_operation_type(op_name)
            if op_name in O1:  # UNARY
                pos_1 = s.pop(-1)
                gen = _operate_one_feature(pos_1, op_func=op_sign, op=op_name)
                if gen is not None:
                    s.append(gen)
                else:
                    s.append(pos_1)
            else:
                pos_1 = s.pop(0)
                pos_2 = s.pop(0)
                gen = _operate_two_features(pos_1, pos_2, op_func=op_sign, op=op_name)
                if gen is not None:
                    s.append(gen)
                else:
                    s.append(pos_1)

    return s[0]


def check_valid(process_seq):
    s = []
    for i in process_seq:
        if i >= (len(operation_set) + 5):  # number
            s.append(f'{i}')
        else:
            op_name = op_map[i]
            if op_name in O1:  # UNARY
                if len(s) < 1:
                    return False
                pos_1 = s.pop(-1)
                s.append(f'({pos_1}, {op_name})')
            else:
                if len(s) < 2:
                    return False
                pos_1 = s.pop(0)
                pos_2 = s.pop(0)
                s.append(f'({pos_1}, {op_name},{pos_2})')
    return len(s) == 1


def converge(seq):
    s1 = []
    s2 = []
    for i in seq:
        if i == '(':
            s1.append(i)
        elif i == ')':
            while True:
                token = s1.pop(-1)
                if token == '(':
                    break
                else:
                    s2.append(token)
        elif i in operation_set:
            while True:
                if len(s1) == 0 or s1[-1] == '(':
                    s1.append(i)
                    break
                else:
                    s2.append(s1.pop(-1))
        else:
            s2.append(i)
    return s2

# def converge_pso(seq):
#     s1 = []
#     s2 = []
#     for i in seq:
#         if i == '(':
#             s1.append(i)
#         elif i == ')':
#             while s1[-1] != '(':
#                 s2.append(s1.pop(-1))
#             s1.pop(-1)
#         elif i in operation_set:
#             while s1[-1] != '(':
#                 s2.append(s1.pop(-1))
#             s1.append(i)
#         else:
#             s2.append(i)
#     return s2

def converge_pso(seq):
    s1 = []
    s2 = []
    for i in seq:
        if i == '(':
            s1.append(i)
        elif i == ')':
            while s1[-1] != '(':
                s2.append(s1.pop())
            s1.pop()
        elif i in operation_set:
            while s1[-1] != '(':
                s2.append(s1.pop())
            s1.append(i)
        else:
            s2.append(i)
    return [s2.pop(0) for i in range(len(s2))]

def split_list(lst, split_value):
    result = []
    temp = []
    for item in lst:
        if item == split_value:
            if temp:
                result.append(temp)
                temp = []
        else:
            temp.append(item)
    if temp:
        result.append(temp)
    return result


if __name__ == '__main__':
    # df = pandas.DataFrame(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    # # op = add_binary(op_map_r['+'], 0, 1) # 0 + 1
    # # op = add_binary(op_map_r['+'], op, 1) # (0+1)+1
    # # op2 = add_unary(op_map_r['cube'], 2) # 2 ^ 3
    # # op = add_binary(op_map_r['+'], op, op2) # ((0+1)+1) + 2^3 = 31
    # # op = add_binary(op_map_r['+'], 0, 1)
    # # op = add_binary(op_map_r['*'], op, 2)
    # # op = add_binary(op_map_r['-'], op, 2) # (0 + 1) x 2 - 2

    # op = add_binary(op_map_r['+'], 0, 1)
    # op = add_binary(op_map_r['/'], op, 2)
    # # op = add_unary(op_map_r['cube'], op)
    # print(op)
    # op = add_binary(op_map_r['-'], op, 2)  # (0 + 1) x 2 - 2
    # op = add_binary(op_map_r['+'], 1, op)  # 1 + (0 + 1) x 2 - 2
    # # op = add_unary(op_map_r['cube'], op)
    # print(op)
    # op_seqs = [int(i) for i in op.split(',')]
    # print('test op records')
    # print(op_seq(df, op_seqs))
    # print('show op records')
    # print(show_ops(op_seqs))
    # c = show_ops(op_seqs)
    # print(c)
    # post_op = converge(show_ops(op_seqs))
    # print(f'trans to post:{post_op}')
    # test_post_op = converge_pso(show_ops(op_seqs))
    # print(f'trans to test post:{test_post_op}')
    print(check_valid(show_ops_r(converge('(1+((2+3)*4)-5)'))))
    # c_ = show_ops_r(post_op)
    # print(f'encode post op as : {c_}')
    # print('test post_op records')
    # print(op_post_seq(df, c_))
    # print(check_valid(c_))
    # print(check_valid([22, 21, 22, 17, 23, 20, 16, 23, 18, 17, 16]))
    # print(check_valid([22, 21, 22, 17, 23, 20, 16, 23, 18, 17, 16, 22]))
    # print(show_ops(c_))
    # print(show_ops([22, 21, 22, 17, 23, 20, 16, 23, 18, 17, 16, 22]))
    # print(check_valid(show_ops_r(converge(show_ops([22, 21, 22, 17, 23, 20, 16, 23, 18, 17, 16])))))

    # df = pandas.read_hdf('NIPS_Code/lstm/utils/data/openml_586.hdf')
    # database = []
    # acc = []
    # with open('tmp/openml_586/STANDALONE.adata', 'rb') as f:
    #     for line in f:
    #         line = line.decode('utf-8')  # 将二进制行解码为字符串
    #         line = [float(item) for item in line.split(',')]
    #         if 21.0 in line and 46.0 in line:
    #             # print(line)
    #             start = line.index(21.0) + 1
    #             end = line.index(46.0)
    #             data = line[start:end]
    #             # print(data)
    #             acc.append(line[end+1])
    #             data_list = data[1:-1]
    #             # 将剩余的元素转换为整数
    #             # data_list = [int(item) for item in data_list if item.isdigit()]
    #             data_list = split_list(data_list, 4.0)
    #             data_list = [[int(item) for item in sublist] for sublist in data_list]
    #             # print(data_list)
    #             database.append(data_list)
    # with open('prompt.txt', 'w') as f:
    # # 保存原始的stdout
    #     original_stdout = sys.stdout
    #     # 重定向stdout到文件
    #     sys.stdout = f
        
    #     for i in range(len(database)):
            
    #         ops_database = database[i]
    #         op_string = '['
    #         for ops in ops_database:
    #             ops = show_ops(ops)
    #             for op in ops:
    #                 if op.isdigit():
    #                     op_string += 'f'+op
    #                 else:
    #                     op_string += op
    #             op_string += ','
    #         op_string = op_string[:-1]
    #         op_string += ']'
    #         print(op_string)
    #             # op = show_ops(ops)
    #             # print(new_string, end=',')

            

    