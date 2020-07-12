import torch
import torchvision
import numpy as np
import csv

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from maml import MAML
from train import adaptation, test
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

testset = miniimagenet("data", ways=5, shots=5, test_shots=15, meta_test=True, download=True)
testloader=BatchMetaDataLoader(testset, batch_size=2, num_workers=4, shuffle=True)
evaliter = iter(testloader)

model_path = './model/model.pth'
model = MAML().to(device)
model.load_state_dict(torch.load(model_path))
loss_fn = torch.nn.CrossEntropyLoss().to(device)

test_loss_log = []
test_acc_log = []

for i in range(1000):
    evalbatch = evaliter.next()
    model.eval()
    testloss, testacc = test(model, evalbatch, loss_fn, lr=0.01, train_step=10, device=device)

    test_loss_log.append(testloss.item())
    test_acc_log.append(testacc)

    print("i {}: test_loss = {:.4f}, test_acc = {:.4f}".format(i, testloss.item(), testacc))

all_result = {'test_loss': test_loss_log, 'test_acc': test_acc_log}

result_path = './log/test'
with open(result_path + '.pkl', 'wb') as f:
    pickle.dump(all_result, f)