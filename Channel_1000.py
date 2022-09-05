# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:16:47 2022

@author: 11543
"""

import numpy as np
from numpy import matlib
import math
import matplotlib.pyplot as plt
import random

Num_AP = 4      #number of access points
local_UE = 10   #maximum number of local user for each AP

#One AP has fixed number of local UE and located randomly in range (101x101)
class Ap(object):
    
    def __init__(self):
        self.UE = local_UE
        self.location = [[np.random.randint(0,102),np.random.randint(0,102)]]
        for i in range(self.UE-1):
            self.locationnext = [[np.random.randint(0,102),np.random.randint(0,102)]]
            self.location = np.append(self.location,self.locationnext,axis = 0)
   
    def __str__(self):
        return f'{self.UE}'

# initialise the location of each UE of each AP    
Ap1 = Ap()
Ap2 = Ap()
Ap3 = Ap()
Ap4 = Ap()

m = Num_AP # total number of APs, which is 4 in this case
n = Ap1.UE + Ap2.UE + Ap3.UE + Ap4.UE #total number of UEs 


# all UEs assigned into one map size(202x202)
for i in range(Ap1.UE):
    Ap1.location[i][1] += 101
for i in range(Ap2.UE):
    Ap2.location[i][0] += 101
    Ap2.location[i][1] += 101
for i in range(Ap4.UE):
    Ap4.location[i][0] += 101
    
def Direction(x0, y0, x, y):
    direct = math.atan((y - y0)/(x - x0))
    if (x-x0) < 0:
        if (y - y0) > 0:
           direct += math.pi
        elif (y - y0) < 0:
            direct -= math.pi
    elif (x-x0)==0 and (y-y0)==0:
        direct = 0
    return direct
    
    
Loc = np.concatenate((Ap1.location, Ap2.location, Ap3.location, Ap4.location), axis=0)# locations of all UEs
Locap = np.array([[51,152],[152,152],[51,51],[152,51]])  
D = np.empty((m, n))# matrix of distance for AP vs UE 
for i in range(n):
    D[0][i] = math.sqrt((Loc[i][0]-51)**2 + (Loc[i][1]-152)**2)
    D[1][i] = math.sqrt((Loc[i][0]-152)**2 + (Loc[i][1]-152)**2)
    D[2][i] = math.sqrt((Loc[i][0]-51)**2 + (Loc[i][1]-51)**2)
    D[3][i] = math.sqrt((Loc[i][0]-152)**2 + (Loc[i][1]-51)**2)

Dap = np.empty((m, m))
for i in range(m):
    Dap[0][i] = math.sqrt((Locap[i][0]-51)**2 + (Locap[i][1]-152)**2)
    Dap[1][i] = math.sqrt((Locap[i][0]-152)**2 + (Locap[i][1]-152)**2)
    Dap[2][i] = math.sqrt((Locap[i][0]-51)**2 + (Locap[i][1]-51)**2)
    Dap[3][i] = math.sqrt((Locap[i][0]-152)**2 + (Locap[i][1]-51)**2)
    
Phase = np.empty((m, n))# matrix of distance for AP vs UE 
for i in range(n):
    Phase[0][i] = Direction(51, 152, Loc[i][0], Loc[i][1])
    Phase[1][i] = Direction(152, 152, Loc[i][0], Loc[i][1])
    Phase[2][i] = Direction(51, 51, Loc[i][0], Loc[i][1])
    Phase[3][i] = Direction(152, 51, Loc[i][0], Loc[i][1])

Phaseap = np.empty((m, m))
for i in range(m):
    Phaseap[0][i] = Direction(51, 152, Locap[i][0], Locap[i][1])
    Phaseap[1][i] = Direction(152, 152, Locap[i][0], Locap[i][1])
    Phaseap[2][i] = Direction(51, 51, Locap[i][0], Locap[i][1])
    Phaseap[3][i] = Direction(152, 51, Locap[i][0], Locap[i][1])
    
#=====================================================================    
D0 = np.empty((1, n))# matrix of distance for AP vs UE 
for i in range(n):
    D0[0][i] = math.sqrt((Loc[i][0])**2 + (Loc[i][1])**2)

Dap0 = np.empty((1, m))
for i in range(m):
    Dap0[0][i] = math.sqrt((Locap[i][0])**2 + (Locap[i][1])**2)
    
Phase0 = np.empty((1, n))# matrix of distance for AP vs UE 
for i in range(n):
    Phase0[0][i] = Direction(0, 0, Loc[i][0], Loc[i][1])

Phaseap0 = np.empty((1, m))
for i in range(m):
    Phaseap0[0][i] = Direction(0, 0, Locap[i][0], Locap[i][1])
    
state = np.zeros((1,(n+m)*2), dtype=float)    
for i in range(m):
    state[0, 2*i] = Dap0[0, i]
    state[0, 2*i+1] = Phaseap0[0, i]
                        
for i in range(n):
    state[0, i*2 + 8] = D0[0, i]
    state[0, i*2 + 8 + 1] = Phase0[0, i]
    
#======================================================================
    

#generate channel with rayleigh fading
def rayleigh (T, N, f_c, speed, n):
    fd = speed*f_c/3e8 # max Doppler shift
    w_M = 2 * np.pi * fd #maximum Doppler frequency shift
    t = np.arange(0, n, 1/T) # time vector. (start, stop, step)
    x = np.zeros(len(t))
    y = np.zeros(len(t))

    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * np.pi
        phi = (np.random.rand() - 0.5) * 2 * np.pi
        x = x + np.random.randn() * np.cos(w_M * t * np.cos(alpha) + phi)
        y = y + np.random.randn() * np.sin(w_M * t * np.cos(alpha) + phi)

    z = (1/np.sqrt(N)) * (x + 1j*y) 
    return z

# generate matrix of path loss
def PL(D):
    L = np.zeros((m, n))
    k0 = 39#PL at 1
    a1 = 2#path loss exponent before
    a2 = 4#path-loss exponent after
    db = 100#break point distance
    for i in range(m):
        for j in range(n):
            d = D[i,j]
            if d <= db:
                L[i,j] = k0 + 10 * a1 * np.log10(d)
            else:
                L[i,j] = k0 + 10 * a2 * np.log10(d) - 10 * np.log10(db ** (a2-a1))
    return L


shadowing = 7 #assume shadowing is 7db          
Loss = PL(D) + shadowing * np.random.randn(m,n)
H_l = np.sqrt(np.power(10,-Loss/10))#loss

H = np.zeros((m, n, 2000), dtype=complex)
for i in range(m):
    for j in range(n):
        H[i, j] = rayleigh(500, 100, 3.5e6, 1, 4)

H_l = np.expand_dims(H_l, axis = 2)
H *= H_l

H_abs = np.abs(H)**2

np.save("H_abs.npy", H_abs)
np.save("D.npy", D)
np.save("Dap.npy", Dap)
np.save("Phase.npy", Phase)
np.save("Phaseap.npy", Phaseap)

np.save("D0.npy", D0)
np.save("Dap0.npy", Dap0)
np.save("Phase0.npy", Phase0)
np.save("Phaseap0.npy", Phaseap0)

np.save("State.npy", state) 