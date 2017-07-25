import numpy as np
import cv2
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.nan)

img = cv2.imread('letters.jpg')
colorImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# plt.imshow(colorImg)
# plt.show()

num_rows = 50
num_test = 12
cell_width = 10
num_train = num_rows - num_test

# Now we split the image to 1300 cells, each 10x10 size
cells = [np.hsplit(row,26) for row in np.vsplit(colorImg,num_rows)]
# plt.imshow(cells[0][0])
# plt.show()

# Make it into a Numpy array. It size will be (26,25,30,30,3)
a = np.array(cells)

# Shuffle data columns
indices = np.random.permutation(len(a)) 
a = a[indices]

# Now we prepare train_data and test_data.
X_train = a[:num_train, :].reshape(-1,cell_width**2).astype(np.float32) # Size = (520,2700)
X_test = a[num_train:, :].reshape(-1,cell_width**2).astype(np.float32) # Size = (130,2700)

# Create labels for train and test data
L = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
y_train = np.tile(L, num_train)# Size = (520,2700)
y_test = np.tile(L, num_test)# Size = (520,2700)

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

for size in [120, 140, 160, 180]:
    training_score = 0
    testing_score = 0
    for i in range(iterations):
        mlp = MLPClassifier(hidden_layer_sizes=(size,size,size),solver='sgd', max_iter=1000)
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
print("MLP cv testing-data score:", best_test_score)
print("Best hidden layer size:", best_size)
print('\n')

mlp = MLPClassifier(hidden_layer_sizes=(best_size, best_size, best_size),solver='sgd', max_iter=1000)
mlp.fit(X_train, y_train)
result = mlp.predict(X_test)

matches = result==y_test
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print (accuracy)
