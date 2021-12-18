'''
This package is for calculating Policy distance.

What is policy distance ? 
Algorithm:
    min_cost = 0
    for each blocks in the nxn square:
        min_cost += (number of different actions between ideal and actual action set) 
    return min_cost
'''
import numpy as np

# State:{選擇的動作，下個STATE}
grid_map = {4:{3:3}, 3:{3:2}, 2:{3:1}, 1:{3:0}, 0:None,
    9:{0:4,3:8}, 8:{0:3,3:7}, 7:{0:2,3:6}, 6:{0:1,3:5}, 5:{0:0},
    14:{0:9,3:13}, 13:{0:8,3:12}, 12:{0:7,3:11}, 11:{0:6,3:10}, 10:{0:5},
    19:{0:14,3:18}, 18:{0:13,3:17}, 17:{0:12,3:16}, 16:{0:11,3:15}, 15:{0:10},
    24:{0:19,3:23},23:{0:18,3:22},22:{0:17,3:21},21:{0:16,3:20}, 20:{0:15}}

class calDis:
    def __init__(self, cost = 0, state = 24):
        self.cost = cost
        self.state = state

def MinCostPolicy5x5(policy):
    '''
    This method is written by 柯秉志
    An implementation via BFS to calculate MinCost
    '''
    total = 0
    for start in range(1,25):
        s = calDis(state=start)
        qu = [s]
        min_cost = 24

        while len(qu):
            s = qu.pop(0)
            idx = s.state
            c1, c2 = s.cost, s.cost

            if(idx == 0):
                min_cost = min(min_cost, s.cost)
                continue
            
            act = policy[idx]
            if act != 0:
                c1 += 1
            if act !=3:
                c2 += 1
            
            next_s = grid_map[idx].get(0,None)
            if next_s is not None:
                tmp = calDis(c1, next_s)
                qu.append(tmp)

            next_s = grid_map[idx].get(3,None)
            if next_s is not None:
                tmp = calDis(c2, next_s)
                qu.append(tmp)
        total+=min_cost
    return total

def MinCostDP5x5(policy: list): #0up  1right 2down 3left
    '''
    This method is written by 徐煜倫
    A faster implementation via DP to calculate MinCost

    actions mapping:
    0: up
    1: right
    2: down
    3: left
    '''
    policy = np.reshape(np.array(policy),(5,5))
    tmp = np.zeros(shape=(5,5),dtype=np.int16)
    for i in range(1,5):
        tmp[0][i] = tmp[0][i-1]+1 if policy[0][1]!=3 else tmp[0][i-1]
        tmp[i][0] = tmp[i-1][0]+1 if policy[1][0]!=0 else tmp[i-1][0]
    for i in range(1,5):
        for j in range(1,5):
            tmp[i][j] = min(tmp[i][j-1],tmp[i-1][j])+1 if policy[i][j]!=3 and policy[i][j]!=0 else min(tmp[i][j-1],tmp[i-1][j])
    return np.sum(tmp)

if __name__=="__main__":
    # Evaluation of MinCostPolicy5x5 & MinCostDP5x5
    policy = [2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,1,1,3]
    import time
    #################BFS################
    start = time.time()
    for _ in range(1000):
        a = MinCostPolicy5x5(policy)
    print("Origin",time.time()-start)
    print(a)
    #################DP#################
    start = time.time()
    for _ in range(1000):
        a = MinCostDP5x5(policy)
    print("DP",time.time()-start)
    print(a)