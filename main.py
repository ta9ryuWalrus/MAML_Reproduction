import torch
import torchvision
import numpy as np
import csv

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from maml import MAML
from train import adaptation
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# dataset

trainset = miniimagenet("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
trainloader = BatchMetaDataLoader(trainset, batch_size=25, num_workers=4, shuffle=True)

testset = miniimagenet("data", ways=5, shots=5, test_shots=15, meta_test=True, download=True)
testloader=BatchMetaDataLoader(testset, batch_size=25, num_workers=4, shuffle=True)

# training

epochs = 100
model = MAML().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
model_path = "./model/"
result_path = "./log/train"

batchiter = iter(trainloader)

for epoch in range(epochs):
    loss_log = []
    acc_log = []

    batch = batchiter.next()
    loss, acc = adaptation(model, optimizer, batch, loss_fn, lr=0.4, train_step=1, train=True, device=device)
    
    loss_log.append(loss.item())
    acc_log.append(acc)
    print("Epoch {}: loss = {:.4f}, acc = {:.4f}".format(epoch, loss.item(), acc))


torch.save(model.state_dict(), model_path + 'model.pth')
all_result = {'train_loss': loss_log, 'train_acc': acc_log}

with open(result_path + '.pkl', 'wb') as f:
    pickle.dump(all_result, f)