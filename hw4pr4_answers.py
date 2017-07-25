import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('boston.csv', header=0)
df = df.iloc[:,:].values
df /= 500

X_train = df[20:, 1:]
y_train = df[20:, 0]
X_test = df[0:20, 1:]

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# Fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

best_test_score = 0
best_train_score = 0
best_size = 0
iterations = 10
print('\n')
print("cv testing scores")
print('-----------------')

for size in [1, 5, 10, 20, 40, 80, 120]:
    training_score = 0
    testing_score = 0
    for i in range(iterations):
        mlp = MLPRegressor(hidden_layer_sizes=(size,size,size),solver='sgd', max_iter=1000)
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            train_test_split(X_train, y_train, test_size=0.25)

        mlp = mlp.fit(cv_data_train, cv_target_train)
        training_score += mlp.score(cv_data_train,cv_target_train)
        testing_score += mlp.score(cv_data_test,cv_target_test)
        
    training_score /= iterations
    testing_score /= iterations

    print("Size", size,":", testing_score)
    if testing_score > best_test_score:
        best_test_score = testing_score
        best_train_score = training_score
        best_size = size

print('\n')
print("MLP cv training-data score:", best_train_score)
print("MLP cv testing-data score:", best_test_score)
print("Best hidden layer size:", best_size)
print('\n')

mlp = MLPRegressor(hidden_layer_sizes=(best_size, best_size, best_size),solver='sgd', max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(predictions)