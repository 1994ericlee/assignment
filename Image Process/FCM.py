# %%
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv2.imread('irabu_zhang1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

w = img.shape[1]
h = img.shape[0]
data = np.asarray(img.transpose([1,0,2]))
resize_data = data.reshape(w*h, 3)

k = 5
# select k random rgb of pixel in the img
random_center = random.sample(resize_data.tolist(), k)

def eucli_dis(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector2-vector1, 2)))

def init_random():
    prob_Matrix = np.random.dirichlet(np.ones(k),size= w*h)
    print(prob_Matrix)
    return prob_Matrix
    
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
   
fuzzy_m = 2
iteration = 200
feature = 3
class_Matrix = np.zeros([450*350])
centroid = np.zeros([k, feature])


prob_Matrix = init_random()

for iter in range (iteration):
    for c in range(k):
        for f in range(feature):
            sum_WP = 0
            sum_weight = 0
            for i in range (len(prob_Matrix)):
                sum_weight = sum_weight + (np.power(prob_Matrix[i,c],fuzzy_m))  
                sum_WP = sum_WP + np.power(prob_Matrix[i, c], fuzzy_m)*resize_data[i, f]
            cc =  sum_WP/sum_weight  
            centroid[c][f] =  cc
    print(centroid) 

    for m in range (len(prob_Matrix)):
        dis_pc = 0
        for j in range (k):
            dis_pc += np.power(1/ eucli_dis(centroid[j, :], resize_data[m, :]), 1/(fuzzy_m-1))
        for j in range (k):
            w = np.power(1/eucli_dis(centroid[j, :], resize_data[m, :]), 1/(fuzzy_m-1) )/ dis_pc
            prob_Matrix[m][j] = w

for i in range(len(resize_data)):    
    cNumber = np.where(prob_Matrix[i] == np.amax(prob_Matrix[i]))
    class_Matrix[i] = cNumber[0]
class_Matrix = class_Matrix.reshape([w, h])    
print(class_Matrix)  


plt.imshow(generate_img(class_Matrix))
plt.show()        
              

# %%
