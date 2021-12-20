import torch
import torch.nn as nn
import torch.nn.functional as F
from grid_world import Gridworld_RandReward_5x5_Env
import numpy as np
import time
import os
import random
from collections import deque


class replay_buffer(object):
    '''
    storing trajectories in deque
    insert: insert a trajectory to the dequeu
    sample: sample a trajectory from the deque
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def insert(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class C51_Net(nn.Module):
    '''
    input: state
    output: a distribution with 51 atoms
    '''

    def __init__(self,  num_actions, atoms, hidden_layer_size=50):
        super(C51_Net, self).__init__()
        self.input_state = 1
        self.atoms = atoms
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.input_state, 32)
        self.fc2 = nn.Linear(32, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_actions*atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x.view(-1, self.num_actions, self.atoms), dim=2)
        return x


class C51_agent():
    def __init__(self, env, vmin=-1, vmax=1, epsilon=0.95, learning_rate=5*0.0001, GAMMA=0.95, batch_size=128, capacity=10000, atoms=51):
        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.n_states = env.observation_space.n  # number of states
        self.n_actions = env.action_space.n  # number of actions
        self.atoms = atoms  # number of atoms
        self.vmin = vmin  # value of min
        self.vmax = vmax  # value of max
        self.count = 0
        self.capacity = capacity
        z_values = torch.linspace(
            self.vmin, self.vmax, self.atoms)  # make a categorigal distribution
        self.z_values = z_values
        self.delta = (self.vmax - self.vmin) / (self.atoms - 1)

        self.buffer = replay_buffer(self.capacity)
        self.batch_size = batch_size

        self.evaluate_net = C51_Net(self.n_actions, self.atoms)
        self.target_net = C51_Net(self.n_actions, self.atoms)
        self.current_distribution = None  # size: actions X atoms (probability)
        self.target_distribution = None  # size: actions X atoms (probability)
        self.Q_s_a = None  # size: Actions (value)
        self.epsilon = epsilon
        self.gamma = GAMMA
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)

    def learn(self):
        '''
        steps:
        ------
        1. update target net by net every 100 times
        2. sample trajectories of batch size
        3. pass memory to net and target net
        4. compute loss with KL-divergence
        5. zero grad
        6. back propagation
        7. optimize by Adam
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        observations, actions, rewards, next_observations, dones = self.buffer.sample(
            self.batch_size)
        with torch.no_grad():
            observation = torch.Tensor(observations)
            next_observation = torch.Tensor(next_observations)
            loss = 0

        self.current_distribution = self.evaluate_net.forward(
            observation)

        self.target_distribution = self.target_net.forward(
            next_observation).detach().cpu()

        a_star = torch.argmax(
            torch.sum(self.target_distribution*self.z_values, dim=2), dim=1)

        self.target_distribution = self.target_distribution.numpy()
        a_star = [int(a) for a in a_star]
        do = []
        for d in dones:
            if d == True:
                do.append(0)
            else:
                do.append(1)
        do = np.array(do)
        T_d = np.zeros([self.batch_size, self.atoms], dtype=np.float32)
        for i in range(self.atoms):
            T_z = np.clip(rewards + do * self.gamma *
                          (self.vmin + i * self.delta), self.vmin, self.vmax)
            b = (T_z - self.vmin) / self.delta
            l = np.floor(b).astype(np.int64)
            u = np.ceil(b).astype(np.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            id = range(self.batch_size)
            T_d[id, l[id]] += self.target_distribution[id,
                                                       a_star, i] * (u[id]-b[id])
            T_d[id, u[id]] += self.target_distribution[id,
                                                       a_star, i] * (b[id]-l[id])
        for b in range(self.batch_size):
            loss += nn.KLDivLoss(size_average=False)(
                self.current_distribution[b][actions[b]].log(), torch.from_numpy(T_d[b]))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        '''
        if random bigger than epsilon:  
            pick the argmax of Q(s,a)
        else:  
            random choose from action space
        '''
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if np.random.uniform() < self.epsilon:
                self.current_distribution = self.evaluate_net.forward(
                    state).squeeze(0).cpu()
                self.Q_s_a = torch.sum(
                    self.current_distribution*self.z_values, dim=1)
                action = int(torch.argmax(self.Q_s_a).numpy())
            else:
                action = np.random.randint(0, self.n_actions)
        return action


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    '''
        steps:
        ------
        1. reset env
        2. done = false
        3. if done is true, go to step 9
        4. choose best action
        5. get feedback from env
        6. store memory
        7. if memory bigger than "exploration" then "learn"
        8. go to next state
        9. record the maximum of Z_value from target net
    '''
    rand_seed = 20
    seed_torch(rand_seed)
    env = Gridworld_RandReward_5x5_Env()
    env.seed(rand_seed)
    agent = C51_agent(env)
    episode = 1000
    epr = 0
    for i in range(episode):
        count = 0
        state = env.reset()
        while True:
            count += 1
            agent.count += 1
            # if agent.count % 10000 == 0:
            #     torch.save(agent.target_net.state_dict(),
            #                './preTrained/gridworld_{}.pth'.format(agent.count/10000))
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            epr = reward + epr*agent.gamma
            agent.buffer.insert(state, int(action), reward,
                                next_state, int(done))
            if agent.count >= 1000:
                agent.learn()
                if done == True or count == 1000 or next_state == None:
                    with open("C51_train.txt", 'a', encoding='utf8') as rf:
                        rf.write("{}: ".format(i) + str(epr) + '\n')
                        rf.close()
                    break
            state = next_state
        print("i: {}, reward: {}, count: {}".format(i, epr, count))

    env.close()
