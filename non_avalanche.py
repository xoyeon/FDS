import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score

# Load data
data = pd.read_csv('FDS/dataset/creditcard.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# Split data into initial train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize ensemble of isolation forest models
n_models = X.shape[1]
models = [IsolationForest(random_state=42, n_jobs=-1) for _ in range(n_models)]

# Perform continual learning
for i in range(n_models):
    # Train the ith model on the first i+1 features
    X_train_partial = X_train.iloc[:, :i+1]
    models[i].fit(X_train_partial)

    # Evaluate the ith model on the first i+1 features
    X_test_partial = X_test.iloc[:, :i+1]
    y_pred_partial = models[i].predict(X_test_partial)

    y_pred_partial[y_pred_partial == 1] = 0
    y_pred_partial[y_pred_partial == -1] = 1
    acc_partial = accuracy_score(y_test, y_pred_partial)
    f1_partial = f1_score(y_test, y_pred_partial)
    print(f"Model {i}: accuracy = {acc_partial:.4f}, f1_score={f1_partial:.4f}")
    
# Combine the outputs of all models to obtain the final prediction
y_pred = np.zeros_like(y_test)
for i in range(n_models):
    X_test_partial = X_test.iloc[:, :i+1]
    y_pred_partial = models[i].predict(X_test_partial)
    y_pred += y_pred_partial
y_pred = np.where(y_pred < 0, 1, 0)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Final model: accuracy = {acc:.4f}, f1_score={f1:.4f}")
