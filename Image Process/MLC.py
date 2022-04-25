#Maximum Likelihood Classification 
# %%
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv2.imread('irabu_zhang1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(1,0,2)
print(img.shape)
data = np.asarray(img.transpose([1,0,2]))

rangeX1 = (0, 174)
rangeY1 = (0, 233)
rangeX2 = (189, 248)
rangeY2 = (259, 320)
rangeX3 = (308, 350)
rangeY3 = (250, 294)
rangeX4 = (341, 350)
rangeY4 = (114, 121)
rangeX5 = (293, 310)
rangeY5 = (153, 177)

train_number = 15
class_number = 5
dim_rgb = 3

classRange = [(rangeX1, rangeY1), (rangeX2, rangeY2),(rangeX3, rangeY3),
              (rangeX4, rangeY4),(rangeX5, rangeY5),]
classRange = np.asarray(classRange)

train_sample_pixel = []
train_sample_rgb = []
train_sample_rgb_inverseCovMatrix = []
train_sample_rgb_detCovMatrix = []

for i in range(class_number):
    sample_pixel = []
    sample_rgb = []
    while len(sample_pixel) < train_number:
        x = random.randrange(classRange[i][0][0],classRange[i][0][1])
        y = random.randrange(classRange[i][1][0],classRange[i][1][1])
        sample_pixel.append((x,y))
        r, g, b = img[x][y]
        sample_rgb.append((r,g,b))
    train_sample_pixel.append(sample_pixel)    
    train_sample_rgb.append(sample_rgb)
    sample_rgb = np.array(sample_rgb)
    sample_r = sample_rgb [:,0]
    sample_g = sample_rgb [:,1]
    sample_b = sample_rgb [:,2]
    sample=  np.array((sample_r,sample_g,sample_b))
    covMatrix = np.cov(sample, bias = True)
    inverseCovMatrix = np.linalg.inv(covMatrix)
    detCovMatrix = np.linalg.det (covMatrix)
    
    train_sample_rgb_inverseCovMatrix.append(inverseCovMatrix)
    train_sample_rgb_detCovMatrix.append(detCovMatrix)
    
train_sample_pixel = np.asanyarray(train_sample_pixel)
train_sample_rgb = np.asanyarray(train_sample_rgb)
train_sample_rgb_mean = np.mean(train_sample_rgb, axis = 1)

pixel = (0,0)
def pixel_classify(pixel):
    pixel_rgb = np.asarray(pixel2rgb(pixel))
    classcode = cal_ml(pixel_rgb)
    return classcode

def pixel2rgb(pixel):
    pixel_rgb = img[pixel[0]][pixel[1]]
    return pixel_rgb

def cal_ml(pixel_rgb):
    flag = 0
    old_prob = -1000000
    for i in range (class_number):
        diff = pixel_rgb - train_sample_rgb_mean[i]
        
        if (prob(i, diff)) > old_prob:
            old_prob = prob(i, diff)
            flag = i
    return flag

def prob(classcode, diff):
    
    constant = -0.5*dim_rgb*np.log(2*np.pi) -0.5*np.log(train_sample_rgb_detCovMatrix[classcode])
    prob =  -0.5 * (np.dot(np.dot(diff, train_sample_rgb_inverseCovMatrix[classcode]),diff.T)) - constant
    return prob

def generate_img():
    new_img = np.ones((450, 350, 3))
    for i in range (450):
        for j in range (350):
            pixel = (i, j)
            new_img[i][j] =  train_sample_rgb_mean[pixel_classify(pixel)]      
    new_img = np.array(new_img, dtype = np.uint8)
    new_img = new_img.transpose(1,0,2)   
    return new_img    
   
plt.imshow(generate_img())
plt.show()

# %%
