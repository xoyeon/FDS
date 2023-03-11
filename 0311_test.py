import pandas as pd
import numpy as np
import training_benchmarks

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from sklearn.model_selection import train_test_split
from avalanche.training import Naive

LEARNING_RATE = 0.001

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model
model = training_benchmarks.BinaryClassification()

# CL Benchmark Creation
dataset = training_benchmarks.data
train_stream = training_benchmarks.TrainData
test_stream = training_benchmarks.TestData

# Prepare for training & testing
optimizer = training_benchmarks.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = training_benchmarks.nn.BCEWithLogitsLoss()

'''# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2,
    eval_mb_size=32, device=device)

# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))'''

'''다른 데이터셋 생성'''
cotinual_data = pd.read_csv('FDS/dataset/ctgan_generated.csv')

features = cotinual_data.drop(['Class'], axis=1).values
labels = np.array(cotinual_data.pop('Class'))

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

train_datasets = []
test_datasets = []
for i in range(x_train.shape[0]):
    train_datasets.append(data=x_train[i:i+1], targets=y_train[i:i+1])
    test_datasets.append(data=x_test[i:i+1], targets=test_stream[i:i+1])

train_exp = train_datasets
test_exp = test_datasets

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