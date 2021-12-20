"""
Gridworld is simple 4 times 4 gridworld from example 4.1 in the book: 
    Reinforcement Learning: An Introduction
@author: pinghsieh
@Most of the code was originally borrowed from: https://github.com/podondra/gym-gridworlds

"""

import gym
from gym import spaces
import numpy as np


class Gridworld_RandReward_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_Env, self).__init__()

        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(15)

        self.gridworld = np.arange(
            self.observation_space.n + 1
        ).reshape((4, 4))
        self.gridworld[-1, -1] = 0

        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                           self.observation_space.n,
                           self.observation_space.n))

        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:-1]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
            ):
                next_row = max(0, min(row + d[0], 3))
                next_col = max(0, min(col + d[1], 3))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                          self.observation_space.n), -1)
        self.R[:, 0] = 0

        # Initialize the state arbitrarily
        self.obs = 1

    def step(self, action):
        next_obs = np.random.choice(
            self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # if next_obs == 0:
        #    reward = 1.0
        # else:
        reward = -2.4 + 4.4 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        self.obs = next_obs
        return next_obs, reward, done, None

    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs

    def render(self):
        return None


class Gridworld_RandReward_3x3_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_3x3_Env, self).__init__()

        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(9)

        self.gridworld = np.arange(
            self.observation_space.n
        ).reshape((3, 3))
        #self.gridworld[-1, -1] = 0

        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                           self.observation_space.n,
                           self.observation_space.n))

        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
            ):
                next_row = max(0, min(row + d[0], 2))
                next_col = max(0, min(col + d[1], 2))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                          self.observation_space.n), -1)
        self.R[:, 0] = 0

        # Initialize the state arbitrarily
        self.obs = 1

    def step(self, action):
        next_obs = np.random.choice(
            self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # if next_obs == 0:
        #    reward = 1.0
        # else:
        reward = -2.4 + 4.4 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        # next_obs = None if next_obs == 0 else next_obs
        self.obs = next_obs
        return next_obs, reward, done, None

    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs

    def render(self):
        return None


class Gridworld_RandReward_4x4_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_4x4_Env, self).__init__()

        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(16)

        self.gridworld = np.arange(
            self.observation_space.n
        ).reshape((4, 4))
        #self.gridworld[-1, -1] = 0

        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                           self.observation_space.n,
                           self.observation_space.n))

        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
            ):
                next_row = max(0, min(row + d[0], 3))
                next_col = max(0, min(col + d[1], 3))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                          self.observation_space.n), -1)
        self.R[:, 0] = 0

        # Initialize the state arbitrarily
        self.obs = 1

    def step(self, action):
        next_obs = np.random.choice(
            self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # if next_obs == 0:
        #    reward = 1.0
        # else:
        reward = -1.2 + 2.2 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        return next_obs, reward, done, None

    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs

    def render(self):
        return None


class Gridworld_RandReward_5x5_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_5x5_Env, self).__init__()

        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # observation is between 0 and 24
        self.observation_space = spaces.Discrete(25)

        self.gridworld = np.arange(
            self.observation_space.n
        ).reshape((5, 5))
        #self.gridworld[-1, -1] = 0

        # state transition matrix(a,s,s_prime)
        self.P = np.zeros((self.action_space.n,
                           self.observation_space.n,
                           self.observation_space.n))

        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
            ):
                next_row = max(0, min(row + d[0], 4))
                next_col = max(0, min(col + d[1], 4))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                          self.observation_space.n), -1)
        self.R[:, 0] = 0

        # Initialize the state arbitrarily
        self.obs = 1

    def step(self, action):
        next_obs = np.random.choice(
            self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # if next_obs == 0:
        #    reward = 1.0
        # else:
        reward = -1.2 + 2.2 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        # next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        return next_obs, reward, done, None

    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs

    def reset_24(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = 24
        return self.obs

    def render(self):
        return None
