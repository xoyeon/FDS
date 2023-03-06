import pandas as pd
import numpy as np

import torch
from torch.nn import BCELoss
from torch.optim import SGD

from sklearn.model_selection import train_test_split
from avalanche.benchmarks.generators import dataset_scenario
from avalanche.models import SimpleMLP, SimpleSequenceClassifier
from avalanche.training import Naive
from avalanche.logging import TextLogger
from avalanche.evaluation.metrics import Accuracy

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CL Benchmark Creation
data = pd.read_csv('./dataset/creditcard.csv')
features = data.drop(['Class'], axis=1).values
labels = np.array(data.pop('Class'))

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1234)

# create the scenario
scenario = dataset_scenario(
    train_dataset_list=[x_train],
    test_dataset_list=[x_test],
    task_labels=labels
)

# model
model = SimpleSequenceClassifier(input_size=30, hidden_size=3, num_layers=1, num_classes=2)

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = BCELoss()

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=1,
    eval_mb_size=32, device=device)

# create the logger
logger = TextLogger(open('log.txt', 'w'))

# create the metric
metric = Accuracy()

# train and evaluate the model on the dataset
for i, train_task in enumerate(scenario.train_stream):
    # train the model on the current task
    cl_strategy.train(train_task, logger=logger)

    # evaluate the model on all tasks so far
    for j, test_task in enumerate(scenario.test_stream[:i+1]):
        results = cl_strategy.eval(test_task, metrics=metric)
        print(f"Accuracy on test task {j+1}: {results[metric]:.4f}")
