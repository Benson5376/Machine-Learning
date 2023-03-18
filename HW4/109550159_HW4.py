import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
# y_test = np.load("y_test.npy")

# 7000 data with 300 features
print(x_train.shape)
print(y_train.shape)
# It's a binary classification problem 
print(np.unique(y_train))
print(x_test.shape)

# Question 1
def cross_validation(x_train, y_train, k):
    data_num = len(x_train)
    X = np.arange(data_num) 
    np.random.shuffle(X)
    kfold_data = []

    idx = 0;
    for i in range(k):
        train_index = []
        val_index = []
        for j in range(int(data_num/k)):
            val_index.append(X[idx])
            idx+=1
        if(i<data_num%k):
            val_index.append(X[idx])
            idx+=1
        
        for j in range(data_num):
            if(X[j]) not in val_index:
                train_index.append(X[j])
        
        kfold_data.append([np.array(train_index), np.array(val_index)])
        #print("Split: %s, Training index: %s, Validation index: %s" %(i+1, train_index, val_index))
    return np.array(kfold_data)

kfold_data = cross_validation(x_train, y_train, k=5)
assert len(kfold_data) == 5 # should contain 10 fold of data
assert len(kfold_data[0]) == 2 # each element should contain train fold and validation fold
assert kfold_data[0][1].shape[0] == 1400 # The number of data in each validation fold should equal to training data divieded by K

# Question 2 
best_score = 0
best_gamma = 0
best_c = 0
list_of_score = []

for c in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]: #
    for gamma in [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005]: #
        total_score = 0
        for i in range(5):
            #represent the set of train and validation of each split
            xt = []
            yt = []
            xv = []
            yv = []
            for j in kfold_data[i][0]:
                xt.append(x_train[j])
            for j in kfold_data[i][0]:
                yt.append(y_train[j])
            for j in kfold_data[i][1]:
                xv.append(x_train[j])
            for j in kfold_data[i][1]:
                yv.append(y_train[j])
            
            clf = SVC(C=c, kernel='rbf', gamma=gamma)
            clf.fit(xt, yt)
            ypred = clf.predict(xv)
            total_score += accuracy_score(ypred, yv)
        score = total_score/5
        list_of_score.append(score) # store the score of each grid, will be used in question3
        
        if(score > best_score):
            best_score = score
            best_gamma = gamma
            best_c = c

# Print the best score and the best parameters.
print("Best score:")
print(best_score)
print("Best gamma:")
print(best_gamma)
print("Best C:")
print(best_c)

# Question 3
list_of_score = np.array(list_of_score)
list_of_score = np.around(list_of_score, decimals=3)
list_of_score = list_of_score.reshape(8, 8)

plt.figure(figsize=(12, 12))
axis_gamma = ["0.0000001", "0.0000005", "0.000001", "0.000005", "0.00001", "0.00005", "0.0001", "0.0005"] #
axis_c = ["0.01", "0.1", "1", "10", "100", "1000", "10000", "100000"] #

plt.imshow(list_of_score, cmap='RdBu', aspect='equal', alpha=1.0, origin='lower', vmin=0.5 , vmax=1.0) 
plt.colorbar()

plt.xticks(np.arange(8), axis_gamma, fontsize=10) 
plt.yticks(np.arange(8), axis_c, fontsize=10) 

for i in range(len(axis_gamma)):
    for j in range(len(axis_c)):
        text = plt.text(j, i, list_of_score[i, j], ha="center", va="center", color="black", fontsize=12)

plt.xlabel("Gamma Parameter", fontsize=10)
plt.ylabel("C Parameter", fontsize=10)
plt.title("Hyperparameter Gridsearch", fontsize=18)
plt.show()


# Question 4
best_model = SVC(C=best_c, kernel='rbf', gamma=best_gamma)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
# print("Accuracy score: ", accuracy_score(y_pred, y_test))






















