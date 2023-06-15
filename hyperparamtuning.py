import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from model import Network
from main import train_dataset, test_dataset, device
import random


X_train = []
X_test = []
y_train = []
y_test = []

for i in range(10000):
    train_data = random.choice(train_dataset)
    test_data = random.choice(test_dataset)
    X_train.append(train_data[0].numpy())
    X_test.append(test_data[0].numpy())
    y_train.append(train_data[1])
    y_test.append(test_data[1])


X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


model = NeuralNetClassifier(
    module=Network,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    max_epochs=10,
    lr=0.001,
    device=device,
)


param_grid = {
    'module__hidden': [128, 256, 512],  # Grid search over different number of hidden neurons
    'module__hidden1': [128, 256, 512],  # Grid search over different number of hidden neurons
    'module__hidden2': [128, 256, 512],  # Grid search over different number of hidden neurons
    'lr': [0.001, 0.01, 0.1],
    'max_epochs': [10, 20, 30],
}


gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', refit=True)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
best_params = gs.best_params_


# accuracy = best_model.score(X_test, y_test)


print("Best Model Parameters:", best_params)
# print("Test Accuracy:", accuracy)
