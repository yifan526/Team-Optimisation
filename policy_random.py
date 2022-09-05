from env.Env5.Env_v3 import MyEnv
import numpy as np


env=MyEnv()
env.reset()
env_info=env.get_env_info()
time_limit=env_info['episode_limit']
n_actions=env_info['n_actions']
n_agents=env_info['n_agents']
sum_rew=0
for i in range(1900):
    actions=np.random.randint(low=0,high=n_actions,size=(n_agents))
    #actions = np.array([10,10,10,10])
    rew,_,_=env.step(actions)
    sum_rew+=rew
print(sum_rew)

a = env.Raverage
M1 = sum(sum(env.Raverage))
M2 = np.percentile(env.Raverage, 5)
M3 = a.var()
np.save("Initial.npy", a)
'''
a = np.zeros((1,40), dtype = float)
a1 = env.Raverage
for i in range(4):
    for j in range(10):
        a[0, i*10 + j] = a1[i,j]
# a = env.Reward_record
b = np.arange(40)
import matplotlib.pyplot as plt
plt.bar(b,a[0,:])
plt.xlabel('User Index')
plt.ylabel('Average Data Rate (kbps/Hz)')
'''

