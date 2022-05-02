#%%
import numpy as np

a = np.arange(27).reshape(3,3,3)
b = np.arange(10,22,2).reshape(2,3)

class_Matrix = np.zeros([3, 3])
for i in range (3):
    for j in range (3):
        min = 99999
        minIndex = 0
        for x in range (2):
            d = np.sqrt(np.sum(np.power(b[x]- a[i,j,:], 2)))
            if d < min:
                min = d
                print(x)
                minIndex = x
        if class_Matrix[i][j] != minIndex:
            class_Matrix[i][j] = minIndex

print(class_Matrix)
for x in range (2):
    m = np.nonzero(class_Matrix == x)
    print(m)
    pointsInCluster = a[m]
    print(pointsInCluster)
    mean = np.mean(pointsInCluster, axis= 0)
    b[x, :] = mean
    print(b)