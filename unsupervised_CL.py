# import necessary libraries
from avalanche.benchmarks import datasets
# from avalanche.benchmarks.generators import nc_scenario_from_tensor_lists
from avalanche.benchmarks.generators import nc_benchmark
# from avalanche.training.strategies import GEM
from avalanche.training import GEM
from avalanche.models import SimpleMLP
from avalanche.logging import TextLogger
from avalanche.evaluation.metrics import Accuracy

# generate the dataset
data = [[1, 2, 0], [2, 3, 1], [3, 4, 0], [4, 5, 1], [5, 6, 0]]

# create the scenario
scenario = nc_benchmark(
    train_data=data[:3],
    test_data=data[3:],
    task_labels=[0, 0, 0, 1, 1],
)

# initialize the model
model = SimpleMLP(input_size=3, num_classes=2)

# create the strategy
strategy = GEM(model, memory_size=1000, train_epochs=1, batch_size=256, 
               optimizer='adam', lr=0.001, 
               lam=0.5, alpha=0.5, use_replay=True)

# create the logger
logger = TextLogger(open('log.txt', 'w'))

# create the metric
metric = Accuracy()

# train and evaluate the model on the dataset
for i, train_task in enumerate(scenario.train_stream):
    # train the model on the current task
    strategy.train(train_task, logger=logger)

    # evaluate the model on all tasks so far
    for j, test_task in enumerate(scenario.test_stream[:i+1]):
        results = strategy.eval(test_task, metrics=metric)
        print(f"Accuracy on test task {j+1}: {results[metric]:.4f}")
