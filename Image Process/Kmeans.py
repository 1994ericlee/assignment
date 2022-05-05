# %%
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv2.imread('irabu_zhang1.bmp')
w = img.shape[1]
h = img.shape[0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = np.asarray(img.transpose([1,0,2]))
resize_data = data.reshape(h*w, 3)

k = 5
# select k random rgb of pixel in the img
random_center = random.sample(resize_data.tolist(), k)

def eucli_dis(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector2-vector1, 2)))

class_Matrix = np.zeros([w, h])
centroid = np.asarray(random_center)
clusterChanged = True
while clusterChanged:
    clusterChanged = False
    for i in range (w):
        for j in range (h):
            minDist = 99999
            minIndex = -1
            for x in range(k):
                distance = eucli_dis([centroid[x]], data[i,j,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = x
            if class_Matrix[i][j] != minIndex : 
                class_Matrix[i][j] = minIndex
                clusterChanged = False
    
    for x in range(k):
        points = np.nonzero(class_Matrix == x)
        pointsInCluster = data[points]
        centroid[x, :] = np.mean(pointsInCluster, axis = 0)
    print(centroid)             
print(class_Matrix)     

def generate_img(class_Matrix):
    new_img = np.ones((w, h, 3))
    for i in range (w):
        for j in range (h):
            pixel = (i, j)
            class_Matrix = np.array(class_Matrix, dtype = np.uint8)
            new_img[i][j] =  centroid[class_Matrix[i][j]]      
    new_img = np.array(new_img, dtype = np.uint8)
    new_img = new_img.transpose(1,0,2)   
    return new_img 

plt.imshow(generate_img(class_Matrix))
plt.show()        
              

# %%
