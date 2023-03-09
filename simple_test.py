import pandas as pd
import numpy as np

import torch
from torch.nn import BCELoss
from torch.optim import SGD

from sklearn.model_selection import train_test_split

from avalanche.models import SimpleMLP
from avalanche.training import Naive

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CL Benchmark Creation
data = pd.read_csv('FDS/dataset/creditcard.csv')
### data.shape : (283726, 31) --> 절반은 141863
features = data.drop(['Class'], axis=1).values
labels = np.array(data.pop('Class'))

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

# model
model = SimpleMLP(num_classes=2, input_size=30, hidden_size=3, hidden_layers=1)

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = BCELoss()


'''다른 데이터셋 생성'''
from avalanche.benchmarks.classic import SplitDataset
from avalanche.benchmarks.utils import AvalancheDataset

train_datasets = []
test_datasets = []
for i in range(x_train.shape[0]):
    train_datasets.append(AvalancheDataset(
        data=x_train[i:i+1], targets=y_train[i:i+1]))
    test_datasets.append(AvalancheDataset(
        data=x_test[i:i+1], targets=y_test[i:i+1]))

train_exp = SplitDataset(train_datasets)
test_exp = SplitDataset(test_datasets)

'''다른 데이터셋 러닝 시작'''
# Continual learning strategy
cl_strategy = Naive(model, optimizer, criterion,
                    train_mb_size=32, train_epochs=2, eval_mb_size=32, device=device)

'''# train and test loop over the stream of experiences
results = []
for train_exp in x_train:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(y_train))'''

# train and test loop over the stream of experiences
for i, train_exp in enumerate(x_train):
    cl_strategy.train(train_exp, task_labels=y_train[i])
    acc = cl_strategy.eval(x_test, y_test)
    print(f"Accuracy after task {i+1}: {acc:.4f}")