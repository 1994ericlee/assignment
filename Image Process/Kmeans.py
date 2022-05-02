# %%
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv2.imread('irabu_zhang1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img.transpose(1,0,2)
data = np.asarray(img.transpose([1,0,2]))
resize_data = data.reshape(350*450, 3)

k = 5
# select k random rgb of pixel in the img
random_center = random.sample(resize_data.tolist(), k)


def eucli_dis(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector2-vector1, 2)))

# def kmeans():
class_Matrix = np.zeros([450, 350])
centroid = np.asarray(random_center)
clusterChanged = True
while clusterChanged:
    clusterChanged = False
    for i in range (450):
        for j in range (350):
            minDist = 99999
            minIndex = -1
            for x in range(k):
                distance = eucli_dis([centroid[x]], data[i,j,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = x
            if class_Matrix[i][j] != minIndex : 
                class_Matrix[i][j] = minIndex
                clusterChanged = True
print(class_Matrix)    
for x in range(k):
    points = np.nonzero(class_Matrix == x)
    pointsInCluster = data[points]
    centroid[x, :] = np.mean(pointsInCluster, axis = 0)
print(centroid)             
# print(kmeans())               
                

# %%
