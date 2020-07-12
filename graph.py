import numpy as np
import pickle
import matplotlib.pyplot as plt

log_path = "./log/train"
graph_path = "./graph/"
with open(log_path + '.pkl', 'rb') as f:
    log = pickle.load(f)

