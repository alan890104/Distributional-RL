#%%
import gym
import QRDQN_DiscreteSpace, QRDQN_ContinuousSpace
import numpy as np
import os
import random
import torch

import grid_world

def seed_torch(env, seed=42):
    env.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

### Discrete environment 
def TrainDiscrete(env_name):
    # env = gym.make(env_name)

    ### for gridworld
    env = grid_world.Gridworld_RandReward_5x5_Env()
    
    ### for fixing random seed
    # seed = 20
    # seed_torch(env, seed)

    n_trains = 2000

    print('Do QR-DQN')
    agent = QRDQN_DiscreteSpace.QR_DQN(env=env, num_quantile=50, learning_rate=1e-3)
    EWMA_reward = None
    for idx in range(n_trains):
        ### score is the summation of the reward per episode
        score = agent.learn(env)
        if score is not None:
            EWMA_reward = 0.06*score + 0.94*EWMA_reward if EWMA_reward is not None else score
            present = 'the '+str(idx)+' episode, reward: '+str(score)+', ewma reward: '+ str(EWMA_reward)
            print(present)

        ### Here to adjust your threshold
        if EWMA_reward>=0.8:
            # agent.save_model(env_name, str(idx))
            break

### Continuous environment 
def TrainContinuous(env_name):
    env = gym.make(env_name)

    ### for fixing random seed
    # seed = 20
    # seed_torch(env, seed)

    n_trains = 3000

    print('Do QR-DQN')
    agent = QRDQN_ContinuousSpace.QR_DQN(env=env, num_quantile=50, learning_rate=1e-3)
    EWMA_reward = 200

    for idx in range(n_trains):
        ### score is the summation of the reward per episode
        score = agent.learn(env)
        if score is not None:
            EWMA_reward = 0.06*score + 0.94*EWMA_reward if EWMA_reward is not None else score
            present = 'the '+str(idx)+' episode, reward: '+str(score)+', ewma reward: '+ str(EWMA_reward)
            print(present)
        
        ### Here to adjust your threshold
        if EWMA_reward >= 195:
            # agent.save_model(env_name, str(idx))
            break


def main():
    ### use the env you need
    # env_name = "CartPole-v0"
    env_name = "Gridworld5x5"

    ### choose the type of env to train
    TrainDiscrete(env_name)
    # TrainContinuous(env_name)

if __name__ == '__main__':
    main()
