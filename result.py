#%%
import numpy as np
import pickle
#%%
log_path = "./result"
with open(log_path + '.pkl', 'rb') as f:
    log = pickle.load(f)
#%%
loss = np.array(log['test_loss'])
acc = np.array(log['test_acc'])

print("loss = {:.4f}, acc = {:.4f}".format(np.mean(loss), np.mean(acc)))

# %%
from scipy.stats import sem, t
from scipy import mean
import pickle
#%%
log_path = "./result"
with open(log_path + '.pkl', 'rb') as f:
    log = pickle.load(f)
loss = log['test_loss']
acc = log['test_acc']

#%%
confidence = 0.95
n = len(acc)
m = mean(acc)
std_err = sem(acc)
h = std_err * t.ppf((1+confidence) / 2, n - 1)
print("acc confidence interval: [{}, {}]".format(m-h, m+h))

# %%
print(m)
print(h)

# %%
