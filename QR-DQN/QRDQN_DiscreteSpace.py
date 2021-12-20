import matplotlib.pyplot as plt
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import replaymemory


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class Network(nn.Module):
    def __init__(self, num_state, num_action, num_quantile):
        nn.Module.__init__(self)

        self.num_state = num_state
        self.num_action = num_action
        self.num_quantile = num_quantile

        ### hidden layer is default 128
        self.input_layer = nn.Linear(1, 128)
        self.layer_output = nn.Linear(128, num_action * num_quantile)

    def forward(self, state):
        if torch.cuda.is_available():
            state = state.cuda()
        state = self.input_layer(state)
        state = F.relu(state)
        state = self.layer_output(state)

        return state.view(-1, self.num_action, self.num_quantile)


class QR_DQN():
    def __init__(self, env, num_quantile, learning_rate):
        ### if the action of ENV has the specified format, adjust the argument of the num_action.
        self.Z = Network(num_state=1, num_action=env.action_space.n, num_quantile=num_quantile)
        self.Z_target = Network(num_state=1, num_action=env.action_space.n, num_quantile=num_quantile)
        self.optimizer = optim.Adam(self.Z.parameters(), learning_rate)

        if torch.cuda.is_available():
            self.Z.cuda()
            self.Z_target.cuda()

        self.step = 0
        self.memory_enough = False

        ### Here is the argument of QRDQN
        self.batch_size = 32

        a = np.arange(num_quantile)[::-1]
        self.tau = torch.Tensor(
            (2*a+1)/(2.0*num_quantile)).view(-1)
        if torch.cuda.is_available():
            self.tau = self.tau.cuda()
        self.num_quantile = num_quantile
        
        self.eps = 0.05
        self.discount = 0.95
        self.replaymemory = replaymemory.replaymemory(10000, 1) # replayermemory(size, state space)
        self.n_iter = 1000

    def get_action(self, state):
        ### with epsilon greedy
        with torch.no_grad():
            action = 0
            if np.random.uniform() >= self.eps:
                action = int(self.Z.forward(state).mean(2).max(1)[1])
            else:
                action = np.random.randint(0, self.Z.num_action)

            return action


    def eps_annealing(self):
        # annealing the epsilon(exploration strategy)
        if self.step <= int(1e+3):
            # linear annealing to 0.9 until million step
            self.eps -= 0.9/1e+3
        elif self.step <= int(1e+4):
            # linear annealing to 0.99 until the end
            self.eps -= 0.09/(1e+4 - 1e+3)

    def calculate_loss(self, samples):
        # theta為訓練用的network得出的訓練結果
        # 取出32個利用batch_state得出的結果[32 x 166 x 2] 取出需要的action的值[32 x 第action個 x 2]
        theta = self.Z(samples['states'])[np.arange(self.batch_size), samples['actions']]
        #print('1',theta)
        with torch.no_grad():
            # target_theta為我們想逼近的最終理想distribution
            Z_nexts = self.Z_target(samples['next_states'])
            Z_next = Z_nexts[np.arange(self.batch_size), Z_nexts.mean(2).max(1)[1]]
            dones = samples['dones'].expand(self.batch_size, self.num_quantile) 
            target_theta = samples['rewards'] + self.discount *(1-dones)* Z_next

        diff = target_theta.t().unsqueeze(-1).detach() - theta
        
        loss = huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()

        loss = loss.transpose(0,1)
        loss = loss.mean(1).sum(-1).mean()

        return loss

    def load_model(self,name,episode):
        self.Z.load_state_dict(torch.load('./preTrained/{}/QRDQN_Actor_{}.pth'.format(name,episode)))
        self.Z_target.load_state_dict(torch.load('./preTrained/{}/QRDQN_TargetActor_{}.pth'.format(name,episode)))

    def save_model(self,name,episode):
        torch.save(self.Z.state_dict(), './preTrained/{}/QRDQN_Actor_{}.pth'.format(name,episode))
        torch.save(self.Z_target.state_dict(), './preTrained/{}/QRDQN_TargetActor_{}.pth'.format(name,episode))

    def learn(self, env):
        state = env.reset()
        score = 0

        for i in range(self.n_iter):
            #self.eps_annealing()
            action = self.get_action(torch.Tensor([state]))
            next_state, reward, done, _ = env.step(action)
            # env.render()

            self.replaymemory.push(
                state, action, next_state, reward, done)

            if done:
                state = env.reset()
            else:
                state = next_state

            ### Select the data for replay buffer
            if len(self.replaymemory) < 1000:
                continue
            self.memory_enough = True

            ### Change the score function if yo need
            score += reward

            samples = self.replaymemory.get_samples(self.batch_size)

            loss = self.calculate_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if self.step % 100 == 0:
                self.Z_target.load_state_dict(self.Z.state_dict())
            self.step += 1
        

            if done or i == self.n_iter-1:
                ### print the max summation of quantiles from all actions
                # with torch.no_grad():
                #     print(i,self.Z(torch.Tensor([env.reset()])).mean(2).max(1)[0])
                break

        return score
