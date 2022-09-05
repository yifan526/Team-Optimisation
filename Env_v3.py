# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:17:58 2022

@author: 11543
"""
import gym
from gym import spaces, core

import numpy as np
from numpy import matlib
import math
import matplotlib.pyplot as plt
import random

m = 4  # total number of APs, which is 4 in this case
local_UE = 10  # maximum number of local user for each AP
n = local_UE * 4  # total number of users#
alphar = 0.01  # rate averaging parametre
lmb = 0.9  # trade-off parameter between R-sum and 5th percentile
T = 500  # number of time slots

H_abs = np.load("env/Env5/H_abs.npy")  # import channel for 2000 time slots
D = np.load("env/Env5/D.npy")
Dap = np.load("env/Env5/Dap.npy")
Phase = np.load("env/Env5/Phase.npy")
Phaseap = np.load("env/Env5/Phaseap.npy")
D0 = np.load("env/Env5/D0.npy")
Dap0 = np.load("env/Env5/Dap0.npy")
Phase0 = np.load("env/Env5/Phase0.npy")
Phaseap0 = np.load("env/Env5/Phaseap0.npy")

'''
H_abs = np.load("H_abs.npy")  # import channel for 2000 time slots
D = np.load("D.npy")
Dap = np.load("Dap.npy")
Phase = np.load("Phase.npy")
Phaseap = np.load("Phaseap.npy")
D0 = np.load("D0.npy")
Dap0 = np.load("Dap0.npy")
Phase0 = np.load("Phase0.npy")
Phaseap0 = np.load("Phaseap0.npy")
'''
'''
def Power_sel(choice):
    if choice == 0:
        power = (10 ** (0 / 10)) / 1000
    elif choice == 1:
        power = (10 ** (5 / 10)) / 1000
    elif choice == 2:
        power = (10 ** (10 / 10)) / 1000
    elif choice == 3:
        power = (10 ** (15 / 10)) / 1000
    elif choice == 4:
        power = (10 ** (20 / 10)) / 1000
    elif choice == 5:
        power = (10 ** (25 / 10)) / 1000
    elif choice == 6:
        power = (10 ** (30 / 10)) / 1000
    elif choice == 7:
        power = (10 ** (35 / 10)) / 1000
    elif choice == 8:
        power = (10 ** (40 / 10)) / 1000
    elif choice == 9:
        power = (10 ** (45 / 10)) / 1000
    elif choice == 10:
        power = (10 ** (50 / 10)) / 1000
    return power
'''
def Power_sel(choice):
    power = (10 ** (5*choice / 10)) / 1000    
    return power

def Station(choice):
    
    if choice >= 0 and choice <= 10:
        UE = -1
        Power = choice
    elif choice >= 11 and choice <= 21:
        UE = -2
        Power = choice - 11
    elif choice >= 22 and choice <= 32:
        UE = -3
        Power = choice - 22
    elif choice >= 33 and choice <= 43:
        UE = -4
        Power = choice - 33
    elif choice >= 44 and choice <= 54:
        UE = -5
        Power = choice - 44
    return UE, Power

    

class Bs:
    # self.P = np.array([np.random.randint(0,6),np.random.randint(0,6),np.random.randint(0,6),np.random.randint(0,6)])
    def Action(self, P, UE, t):  # 输入P : np array of int(4x1),每个值在0-5, UE (4x1)array, 值在0-9
        self.action = np.array([Power_sel(P[0]), Power_sel(P[1]), Power_sel(P[2]), Power_sel(P[3])])
        
        self.R_max = np.zeros((m, 1), dtype=float)
        for i in range(m):
            self.sig = H_abs[i, UE[i] + 10 * i, t] * self.action[i]
            self.sig_sum = 0
            for j in range(m):
                self.sig_sum += H_abs[j, UE[i] + 10 * i, t] * self.action[j]
            self.noise = self.sig_sum - self.sig
            self.R_max[i] = np.log2(1 + self.sig / self.noise)

        return self.R_max


class MyEnv:

    def __init__(self):
        self.AP = Bs()
        self.UE = np.array([0, 0, 0, 0])
        self.Power = np.array([0, 0, 0, 0])
        self.t = 0
        #self.Raverage = np.zeros((m, local_UE), dtype=float)
        self.Raverage = np.load("Initial.npy")
        #self.Raverage_long = np.zeros((m, local_UE), dtype=float)
        self.Rmax_record = np.zeros((m, local_UE, T), dtype=float)
        self.Reward = np.zeros((m, 1), dtype=float)
        self.Reward_record = np.zeros((1,T),dtype = float)
        self.record = np.zeros((1,T),dtype = float)
        #self.obs = np.zeros((m, (n + m) * 2 + 1), dtype=float)
        self.obs = self.Raverage
        self.obs_dim = 10
        self.state_shape = 129
        self.n_agents = 4
        self.n_actions = 55
        self.time_limit=500

    def reset(self):
        #self.Raverage = np.zeros((m, local_UE), dtype=float)  # for updating, assume each t interval is 0.01s
        self.Raverage = np.load("Initial.npy")
        #self.Raverage_long = np.zeros((m, local_UE),
                                      #dtype=float)  # long term average rate for updating, assume each t interval is 0.002s
        self.Rmax_record = np.zeros((m, local_UE, T), dtype=float)  # record of maximum data rate
        self.Reward_record = np.zeros((1,T),dtype = float)
        self.UE = np.array([0, 0, 0, 0])
        
        self.t = 0

        # reset obs
        self.obs = self.Raverage
# =============================================================================
#         self.obs = np.zeros((m, (n + m) * 2 + 1), dtype=float)
#         for i in range(m):
#             for j in range(m):
#                 self.obs[i, 2 * j] = Dap[i, j]
#                 self.obs[i, 2 * j + 1] = Phaseap[i, j]
# 
#         for i in range(m):
#             self.obs[:, self.UE[i] * 2 + i * 10 * 2 + 8] = D[:, self.UE[i] + i * 10]
#             self.obs[:, self.UE[i] * 2 + i * 10 * 2 + 8 + 1] = Phase[:, self.UE[i] + i * 10]
#             
#         for i in range(m):
#             self.obs[i,-1] = self.t
# =============================================================================

        # reset state
        self.state = np.zeros((1, (n + m) * 2 + n + 1), dtype=float)
        for i in range(m):
            self.state[0, 2 * i] = Dap0[0, i]
            self.state[0, 2 * i + 1] = Phaseap0[0, i]

        for i in range(m):
            self.state[0, self.UE[i] * 2 + i * 10 * 2 + 8] = D0[0, self.UE[i] + i * 10]
            self.state[0, self.UE[i] * 2 + i * 10 * 2 + 8 + 1] = Phase0[0, self.UE[i] + i * 10]
            
            self.state[0,-1] = self.t

        

        return self.obs, self.state

    def step(self, P):  # input P eg: P = np.array([5,5,5,5])
        self.weight = 1 / self.Raverage
        self.UE[0], self.Power[0] = Station(P[0])
        self.UE[1], self.Power[1] = Station(P[1])
        self.UE[2], self.Power[2] = Station(P[2])
        self.UE[3], self.Power[3] = Station(P[3])
        
        for i in range(m):
            self.UE[i] = np.argsort(self.weight[i,:])[self.UE[0]]
        
        self.state = np.zeros((1, (n + m) * 2), dtype=float)
        for i in range(m):
            self.state[0, 2 * i] = Dap0[0, i]
            self.state[0, 2 * i + 1] = Phaseap0[0, i]

        for i in range(m):
            self.state[0, self.UE[i] * 2 + i * 10 * 2 + 8] = D0[0, self.UE[i] + i * 10]
            self.state[0, self.UE[i] * 2 + i * 10 * 2 + 8 + 1] = Phase0[0, self.UE[i] + i * 10]
#RRRRRRRRRRRRRRRR            
        # calculate Rmax
        self.Rmax = self.AP.Action(self.Power, self.UE, self.t)
        
        self.Rmax1 = np.zeros((1, n), dtype=float)
        for i in range(m):
            self.Rmax_record[i, self.UE[i], self.t] = self.Rmax[i]
            self.Rmax1[0, self.UE[i] + i * 10] = self.Rmax[i]

        self.state = np.append(self.state, self.Rmax1, axis=1)
        self.state = np.append(self.state, self.t)
        self.state = self.state.reshape(1, len(self.state))
        state = self.state

        # calculate Raverage & weight
        self.tnew = self.t + 1
        if self.t != 0:            
            for i in range(m):
                for j in range(local_UE):
                    if j != self.UE[i]:
                        self.Raverage[i, j] = self.Raverage[i, j] * (self.t) / self.tnew
                self.Raverage[i, self.UE[i]] = (self.Raverage[i, self.UE[i]] * (self.t) + self.Rmax[i]) / self.tnew
        #self.weight = 1 / self.Raverage

        # calculate Raverage-long term & weight-long term
        
# =============================================================================
#         for i in range(m):
#             for j in range(local_UE):
#                 self.Raverage_long[i, j] = (1 - alphar) * self.Raverage_long[i, j] + alphar * self.Rmax_record[
#                     i, j, self.t]
#         self.weightl = 1 / self.Raverage_long
# =============================================================================

        # calculate reward(4x1) of current state
        #for i in range(m):
            #self.Reward[i] = self.weight[i, self.UE[i]]** lmb * self.Rmax[i]
        #reward = sum(self.Reward)
        #reward = self.Rmax.mean()
        
# =============================================================================
#         if reward <= 25:
#             reward = 0
# =============================================================================
        
        #self.Reward_record[0,self.t] = reward
        # calculate user selection of next time slot
        '''
        for i in range(m):
            self.UE[i] = np.argmax(self.weight[i, :])
        user = self.UE
        '''
        '''
        for i in range(m):
            if self.UE[i] == 9:
                self.UE[i] = 0
            else:
                self.UE[i] += 1
        '''
        
# =============================================================================
#         reward = 1/self.Rmax.var()
#         self.Reward_record[0,self.t] = reward
# =============================================================================
        '''
        M = np.array([self.Raverage[0,:].mean(), self.Raverage[1,:].mean(), self.Raverage[2,:].mean(), self.Raverage[3,:].mean()])
        reward = min(M) 
        '''
        for i in range(m):
            self.Reward[i] = (self.weight[i, self.UE[i]]** 0.9) * self.Rmax[i]
        reward = self.Reward.mean()
        
        #reward = 1/self.Raverage.var()
        self.Reward_record[0,self.t] = reward
        #reward = 0
# =============================================================================
#         if self.t !=0:
#             if reward > self.Reward_record[0,self.t-1]:
#                 reward = 10
# =============================================================================
        self.record[0,self.t] = reward
        self.t += 1

# =============================================================================
#         self.obs = np.zeros((m, (n + m) * 2 + 1), dtype=float)
#         for i in range(m):
#             for j in range(m):
#                 self.obs[i, 2 * j] = Dap[i, j]
#                 self.obs[i, 2 * j + 1] = Phaseap[i, j]
# 
#         for i in range(m):
#             self.obs[:, self.UE[i] * 2 + i * 10 * 2 + 8] = D[:, self.UE[i] + i * 10]
#             self.obs[:, self.UE[i] * 2 + i * 10 * 2 + 8 + 1] = Phase[:, self.UE[i] + i * 10]
#         
#         for i in range(m):
#             self.obs[i,-1] = self.t
# =============================================================================
        self.obs = self.Raverage
        new_observation = self.obs

        # DONE when finish 2000 time slots
        done = False
        if self.t == (T):
            done = True
        
        #M = np.array([self.Raverage[0,:].mean(), self.Raverage[1,:].mean(), self.Raverage[2,:].mean(), self.Raverage[3,:].mean()])
        #reward = 1/M.var()
        #reward = 1 / self.Rmax.var()
        #reward = np.percentile(self.Raverage, 5)
        #reward = self.Rmax.mean()
        #reward = sum(sum(self.Reward))
        return  reward, done,{}

    def getstate(self):
        return self.state

    def get_env_info(self):
        output_dict = {}
        output_dict['n_actions'] = self.n_actions
        output_dict['obs_shape'] = self.obs_dim
        output_dict['n_agents'] = self.n_agents
        output_dict['state_shape'] = self.state_shape
        output_dict['episode_limit'] = self.time_limit

        output_dict['n_enemy'] = 5
        output_dict['p_state'] = self.n_actions

        return output_dict

    def get_state(self):
        return self.state.flatten()

    def get_obs(self):
        """ Returns all agent observations in a list """

        return self.obs

    def get_obs_size(self):
        """ Returns the shape of the observation """

        return self.obs_dim

    def get_state_size(self):
        """ Returns the shape of the state"""

        return self.state_shape
    def get_p_state(self):
        p_state = np.ones((self.n_agents, self.n_actions))

        return p_state
    def get_avail_agent_actions(self, id):
        avail_actions = np.ones((self.n_agents, self.n_actions))

        return avail_actions