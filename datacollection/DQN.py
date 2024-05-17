import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Net1(nn.Module):

    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class Net2(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DuelingNet1(nn.Module):

    def __init__(self, STATE_DIM, ACTION_DIM):
        super(DuelingNet1, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, 1)
        self.out.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(STATE_DIM, 100)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(100, 1)
        self.out2.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        a = torch.cat((state, action), axis=-1)
        a = self.fc1(a)
        a = F.relu(a)
        a = self.out(a)
        v = self.fc2(state)
        v = F.relu(v)
        v = self.out2(v)
        return a + v


class DuelingNet2(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(DuelingNet2, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        self.out_2 = nn.Linear(100, 1)
        self.out_2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        a = self.out(x)
        v = self.out_2(x)
        action_value = a + v
        return action_value


class DQN1(object):

    def __init__(self, STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8):
        self.eval_net, self.target_net = Net1(STATE_DIM, ACTION_DIM), Net1(
            STATE_DIM, ACTION_DIM)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, STATE_DIM * 2 + ACTION_DIM *
                                2 + 1))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.STATE_DIM = STATE_DIM
        self.ACTION_DIM = ACTION_DIM
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.BATCH_SIZE = BATCH_SIZE

    def get_q_value(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        return self.eval_net(torch.cat((state, action)))

    def get_q_value_next_state(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        return self.target_net(torch.cat((state, action)))

    def store_transition(self, s, a, r, s_, a_):
        transition = np.hstack((s, a, [r], s_, a_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.STATE_DIM])
        b_a = torch.LongTensor(b_memory[:, self.STATE_DIM:self.STATE_DIM +
                                                          self.ACTION_DIM])
        b_r = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.
                                ACTION_DIM:self.STATE_DIM + self.ACTION_DIM + 1])
        b_s_ = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.
                                 ACTION_DIM + 1:self.STATE_DIM * 2 + self.ACTION_DIM + 1])
        b_a_ = torch.LongTensor(b_memory[:, -self.ACTION_DIM:])
        net_input = torch.cat((b_s, b_a), axis=1)
        q_eval = self.eval_net(net_input)
        net_input_ = torch.cat((b_s_, b_a_), axis=1)
        q_next = self.target_net(net_input_).detach()
        q_target = b_r + self.GAMMA * q_next.view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQN2(object):

    def __init__(self, N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        self.eval_net, self.target_net = Net2(N_STATES, N_ACTIONS), Net2(
            N_STATES, N_ACTIONS)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.BATCH_SIZE = BATCH_SIZE
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY

    def choose_next_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        action_value = self.target_net.forward(x)
        action = torch.max(action_value, 1)[1].data.numpy()
        action = action[0]
        return action

    def choose_action(self, x, steps_done):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        if np.random.uniform() > eps_threshold:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES +
                                                              2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQN1(DQN1):

    def __init__(self, STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8):
        super().__init__(STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                         TARGET_REPLACE_ITER, GAMMA, EPSILON, LR, BATCH_SIZE)

    def get_q_value_next_state(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        return self.eval_net(torch.cat((state, action)))


class DDQN2(DQN2):

    def __init__(self, N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        super().__init__(N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                         TARGET_REPLACE_ITER, GAMMA, EPSILON, LR, BATCH_SIZE, EPS_START,
                         EPS_END, EPS_DECAY)

    def choose_next_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        action_value = self.eval_net.forward(x)
        action = torch.max(action_value, 1)[1].data.numpy()
        action = action[0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


class DuelingDQN1(DQN1):

    def __init__(self, STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8):
        super().__init__(STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                         TARGET_REPLACE_ITER, GAMMA, EPSILON, LR, BATCH_SIZE)
        self.eval_net, self.target_net = DuelingNet1(STATE_DIM, ACTION_DIM
                                                     ), DuelingNet1(STATE_DIM, ACTION_DIM)

    def get_q_value(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        return self.eval_net(state, action)

    def get_q_value_next_state(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        return self.target_net(state, action)

    def store_transition(self, s, a, r, s_, a_):
        transition = np.hstack((s, a, [r], s_, a_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.STATE_DIM])
        b_a = torch.LongTensor(b_memory[:, self.STATE_DIM:self.STATE_DIM +
                                                          self.ACTION_DIM])
        b_r = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.
                                ACTION_DIM:self.STATE_DIM + self.ACTION_DIM + 1])
        b_s_ = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.
                                 ACTION_DIM + 1:self.STATE_DIM * 2 + self.ACTION_DIM + 1])
        b_a_ = torch.LongTensor(b_memory[:, -self.ACTION_DIM:])
        q_eval = self.eval_net(b_s, b_a)
        q_next = self.target_net(b_s_, b_a_).detach()
        q_target = b_r + self.GAMMA * q_next.view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DuelingDQN2(DQN2):

    def __init__(self, N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        super().__init__(N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                         TARGET_REPLACE_ITER, GAMMA, EPSILON, LR, BATCH_SIZE, EPS_START,
                         EPS_END, EPS_DECAY)
        self.eval_net, self.target_net = DuelingNet2(N_STATES, N_ACTIONS
                                                     ), DuelingNet2(N_STATES, N_ACTIONS)


class DuelingDDQN1(DuelingDQN1):

    def __init__(self, STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8):
        super().__init__(STATE_DIM, ACTION_DIM, MEMORY_CAPACITY,
                         TARGET_REPLACE_ITER, GAMMA, EPSILON, LR, BATCH_SIZE)
        self.eval_net, self.target_net = DuelingNet1(STATE_DIM, ACTION_DIM
                                                     ), DuelingNet1(STATE_DIM, ACTION_DIM)

    def get_q_value_next_state(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        return self.eval_net(state, action)


class DuelingDDQN2(DDQN2):

    def __init__(self, N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                 TARGET_REPLACE_ITER=100, GAMMA=0.99, EPSILON=0.95, LR=0.01,
                 BATCH_SIZE=8, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        super().__init__(N_STATES, N_ACTIONS, MEMORY_CAPACITY,
                         TARGET_REPLACE_ITER, GAMMA, EPSILON, LR, BATCH_SIZE, EPS_START,
                         EPS_END, EPS_DECAY)
        self.eval_net, self.target_net = DuelingNet2(N_STATES, N_ACTIONS
                                                     ), DuelingNet2(N_STATES, N_ACTIONS)
