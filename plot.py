import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


dir = 'logs/ingolstadt21/a3c_seed_0'
dir_2 = 'logs/ingolstadt21/a3c_seed_4'
name = 'a3c'
obj = 'total_stopped'

a = []

for i in range(0,10):
        dir = f'logs/ingolstadt21/a3c_seed_{i}'
        num = len(os.listdir(dir))
        tmp = []
        for j in range(1, num+1):
                df = pd.read_csv(dir + f'/a3c_conn0_run{j}.csv')
                reward = np.mean(df[obj])
                tmp.append(reward)
        a.append(tmp)


b = list(zip(*zip(*a)))

avg = np.mean(b, axis=0)
var = np.var(b,axis=0)

x= np.array([i for i in range(len(avg))])

avg += var

plt.rcParams["figure.figsize"] = (10, 6)
fig, ax = plt.subplots()
# ax.set_ylim([-0.05, 0.5])

ax.plot(avg, label='A3C')
ax.fill_between(range(len(avg)), avg - var, avg + var, alpha=0.35)
plt.legend()
plt.show()
