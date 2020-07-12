#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt

#%%
log_path = "./log"
graph_path = "./graph/"
with open(log_path + '.pkl', 'rb') as f:
    log = pickle.load(f)

#%%
plt.figure()
plt.plot(log['train_loss'], label='train loss')
plt.plot(log['test_loss'], label='test loss')
plt.legend()
plt.show()