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

epochs = 1000
model = MAML().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
model_path = "./model/"
result_path = "./log/train"

trainiter = iter(trainloader)
evaliter = iter(testloader)

train_loss_log = []
train_acc_log = []
test_loss_log = []
test_acc_log = []

for epoch in range(epochs):
    # train
    trainbatch = trainiter.next()
    model.train()
    loss, acc = adaptation(model, optimizer, batch, loss_fn, lr=0.4, train_step=1, train=True, device=device)
    
    train_loss_log.append(loss.item())
    train_acc_log.append(acc)

    # test
    evalbatch = evaliter.next()
    model.eval()
    testloss, testacc = adaptation(model, optimizer, loss_fn, lr=0.4, train_step=1, train=False, device=device)

    test_loss_log.append(testloss)
    test_acc_log.append(testacc)

    print("Epoch {}: train_loss = {:.4f}, train_acc = {:.4f}, test_loss = {:.4f}, test_acc = {:.4f}".format(epoch, loss.item(), acc, testloss.item(), testacc))


torch.save(model.state_dict(), model_path + 'model.pth')
all_result = {'train_loss': loss_log, 'train_acc': acc_log}

with open(result_path + '.pkl', 'wb') as f:
    pickle.dump(all_result, f)