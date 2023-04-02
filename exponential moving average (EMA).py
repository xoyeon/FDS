!pip install pytorch-tabnet
!pip install torch-ema

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.utils import create_explain_matrix
from torch_ema import ExponentialMovingAverage

# Load data
data = pd.read_csv("creditcard.csv")

# Split data into labeled and unlabeled
X_train, X_test, y_train, y_test = train_test_split(data.drop("Class", axis=1), data["Class"], test_size=0.2, random_state=42)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Convert data to numpy arrays
X_labeled = X_labeled.to_numpy()
X_unlabeled = X_unlabeled.to_numpy()
y_labeled = y_labeled.to_numpy()

# Pretraining with EMA
pretrainer = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type='entmax',
    scheduler_params=dict(max_lr=0.05, steps_per_epoch=100),
    scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
    epsilon=1e-5,
    seed=42,
    device_name='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=1
)

# Create EMA object
ema = ExponentialMovingAverage(pretrainer, decay=0.995)

# Train pretrainer
pretrainer.fit(
    unlabeled_data=X_unlabeled,
    eval_set=[X_test],
    pretraining_ratio=0.8,
    max_epochs=3,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=True,
    on_epoch_end=lambda: ema.update(),  # Update EMA at the end of each epoch
)

# Create TabNet model and fine-tune on labeled data
clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    mask_type='entmax',
    epsilon=1e-5,
    seed=42,
    verbose=1
)

# Load EMA weights into pretrainer
ema.apply_shadow()

# Fine-tune on labeled data with EMA
clf.fit(
    X_labeled=X_labeled,
    y=y_labeled,
    eval_set=[(X_labeled, y_labeled), (X_test, y_test)],
    eval_name=["train", "valid"],
    max_epochs=50,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=True,
    on_epoch_end=lambda: ema.update(),  # Update EMA at the end of each epoch
)

# Get explain matrix for test data
explain_matrix = create_explain_matrix(clf, X_test)

# Evaluate on test data
clf.predict(X_test)
