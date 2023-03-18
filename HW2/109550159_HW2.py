import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 1. Compute the mean vectors mi, (i=1,2) of each 2 classes 

class1 = []
class2 = []

for i in range(len(x_train)):
    if(y_train[i] == 0):
        class1.append(x_train[i])
    else:
        class2.append(x_train[i])

m1 = 0
m2 = 0

for i in range(len(class1)):
    m1 += class1[i]

for i in range(len(class2)):
    m2 += class2[i]

m1 = m1/len(class1)
m2 = m2/len(class2)

print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")

# 2. Compute the Within-class scatter matrix SW
sum_c1 = 0
sum_c2 = 0

for i in range(len(class1)):
    sum_c1 += np.outer((class1[i] - m1), (class1[i] - m1))

for i in range(len(class2)):
    sum_c2 += np.outer((class2[i] - m2), (class2[i] - m2))

sw = sum_c1 + sum_c2

print(f"Within-class scatter matrix SW: {sw}")

# 3. Compute the Between-class scatter matrix SB
sb = np.outer((m2 - m1), (m2 - m1))
print(f"Between-class scatter matrix SB: {sb}")

# 4. Compute the Fisher’s linear discriminant
sw_inv = np.linalg.inv(sw)
w = np.dot(sw_inv, (m2 - m1))
normalized_w0 = w[0]/np.sqrt(w[0]**2 + w[1]**2)
normalized_w1 = w[1]/np.sqrt(w[0]**2 + w[1]**2)
w[0] = normalized_w0
w[1] = normalized_w1
print(f"Fisher’s linear discriminant: {w}")

# 5.Projection and KNN
K = 5
y_pred = np.zeros((1250, ))
proj_train = np.zeros((3750, ))
proj_test = np.zeros((1250, ))
b = 0

# project x_train and x_test to the boundary
for i in range(len(x_train)):
    proj_train[i] = np.dot(x_train[i], w)

for i in range(len(x_test)):
    proj_test[i] = np.dot(x_test[i], w)

for i in range(len(proj_test)):
    distance = np.zeros((3750, 2))
    
    # Compute the difference of the inner product and store the index 
    for j in range(len(proj_train)):
        Diff = abs(proj_test[i] - proj_train[j])
        distance[j][0] = Diff
        distance[j][1] = j
        
    # Sort the difference
    sorted_distance_and_index = sorted(distance, key=lambda x: x[0])
  
    # Pick up the first k term
    K_nearest_distance_and_index = sorted_distance_and_index[:K]
    
    # Pick up the label of first k term
    K_nearest_label = np.zeros((5, ))
    for j in range(K):
        index = K_nearest_distance_and_index[j][1]
        K_nearest_label[j] = y_train[int(index)]
        
    # Give label to the prediction
    num_of_0 = 0
    num_of_1 = 0
    for j in range(K):
        if (K_nearest_label[j] == 0):
            num_of_0 += 1
        else:
            num_of_1 += 1
    if(num_of_1 > num_of_0):
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# Check accuracy       
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of test-set {acc}")

# 6. Plot the graph
class1_x_axis = np.zeros(3750, )
class1_y_axis = np.zeros(3750, )
class2_x_axis = np.zeros(3750, )
class2_y_axis = np.zeros(3750, )

for i in range(len(x_train)):
    if(y_train[i]==0):
        class1_x_axis[i] = x_train[i][0]
        class1_y_axis[i] = x_train[i][1]
    else:
        class2_x_axis[i] = x_train[i][0]
        class2_y_axis[i] = x_train[i][1]

plt.figure(figsize = (8.5,8))       
plt.title(f"Projection Line: w={w[1]/w[0]}, b={b}")
plt.scatter(class1_x_axis, class1_y_axis, color = 'r', s = 1)
plt.scatter(class2_x_axis, class2_y_axis, color = 'b', s = 1)
 
projx_red = np.zeros((3750, ))
projy_red = np.zeros((3750, ))
projx_blue =np.zeros((3750, ))
projy_blue = np.zeros((3750, ))

for i in range(len(x_train)):
    if(y_train[i]==0):
        projx_red[i] = class1_x_axis[i] - w[1] * (w[1]*class1_x_axis[i] - w[0]*class1_y_axis[i] + w[0]*b)/(w[1]**2 + w[0]**2)
        projy_red[i] = (w[1]/w[0])*projx_red[i] + b
    else:
        projx_blue[i] = class2_x_axis[i] - w[1] * (w[1]*class2_x_axis[i] - w[0]*class2_y_axis[i] + w[0]*b)/(w[1]**2 + w[0]**2)
        projy_blue[i] = (w[1]/w[0])*projx_blue[i] + b

plt.scatter(projx_blue, projy_blue, color = 'b', s = 1)
plt.scatter(projx_red, projy_red, color = 'r', s = 1)

for i in range(3750):
    plt.plot([class1_x_axis[i], projx_red[i]], [class1_y_axis[i], projy_red[i]], '-', color="red", lw = 0.3)
    
for i in range(3750):
    plt.plot([class2_x_axis[i], projx_blue[i]], [class2_y_axis[i], projy_blue[i]], '-', color="blue", lw=0.3)
   
plt.plot([1.5,-2.2], [1.5*w[1]/w[0], (-2.2)*w[1]/w[0]], '-', color="purple")
    
    
    




