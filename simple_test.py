import pandas as pd
import numpy as np

import torch
# from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from torch.optim import SGD

# from avalanche.benchmarks.classic import PermutedMNIST
from sklearn.model_selection import train_test_split
# from avalanche.models import SimpleMLP
from avalanche.models import SimpleSequenceClassifier, MTSimpleSequenceClassifier
from avalanche.training import Naive
# from avalanche.training import ObjectDetectionTemplate

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CL Benchmark Creation
# perm_mnist = PermutedMNIST(n_experiences=3)
data = pd.read_csv('FDS/dataset/inflearn_creditcard.csv')

features = data.drop(['Class'], axis=1).values
labels = np.array(data.pop('Class'))

# train_stream = perm_mnist.train_stream
# test_stream = perm_mnist.test_stream
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

# model
## model = SimpleMLP(num_classes=10)
model = SimpleSequenceClassifier(input_size=30 , hidden_size=3 , n_classes=2)

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# criterion = CrossEntropyLoss()
criterion = BCELoss()

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2,
    eval_mb_size=32, device=device)

# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))   