#Maximum Likelihood Classification 
# %%
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv2.imread('irabu_zhang1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = np.asarray(img)

rangeY1 = (0, 174)
rangeX1 = (0, 233)
rangeY2 = (189, 248)
rangeX2 = (259, 320)
rangeY3 = (308, 350)
rangeX3 = (250, 294)
rangeY4 = (341, 350)
rangeX4 = (114, 121)
rangeY5 = (293, 310)
rangeX5 = (153, 177)

pixel_number = 10
class_number = 5
d = 3

classRange = [(rangeX1, rangeY1), (rangeX2, rangeY2),(rangeX3, rangeY3),
              (rangeX4, rangeY4),(rangeX5, rangeY5),]
classRange = np.asarray(classRange)

train_sample_pixel = []
train_sample_rgb = []
for i in range(class_number):
    randPixel = []
    sample_rgb = []
    while len(randPixel) < pixel_number:
        x = random.randrange(classRange[i][0][0],classRange[i][0][1])
        y = random.randrange(classRange[i][1][0],classRange[i][1][1])
        randPixel.append((x,y))
        r, g, b = img[x][y]
        sample_rgb.append((r,g,b))
    train_sample_pixel.append(randPixel)    
    train_sample_rgb.append(sample_rgb)
train_sample_pixel = np.asanyarray(train_sample_pixel)
train_sample_rgb = np.asanyarray(train_sample_rgb)
    
train_sample_rgb_mean = np.mean(train_sample_rgb, axis = 1)

pixel = (0,0)
def pixel_classify(pixel):
    pixel_rgb = np.asarray(pixel2rgb(pixel))
    
    classcode = cal_ml(pixel_rgb)
    return classcode

def pixel2rgb(pixel):
    pixel_rgb = img[pixel[1]][pixel[0]]
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
    
    r = train_sample_rgb[classcode][:, 0]
    g = train_sample_rgb[classcode][:, 1]
    b = train_sample_rgb[classcode][:, 2]
    
    sample = np.array([r, g, b])
    covMatrix = np.cov(sample, bias = True)
    inverseCovMatrix = np.linalg.inv(covMatrix)
    detCovMatrix = np.linalg.det (covMatrix)
    constant = -0.5*d*np.log(2*np.pi) -0.5*np.log(detCovMatrix)
    prob =  -0.5 * (np.dot(np.dot(diff, inverseCovMatrix),diff.T)) - constant
    return prob

print(pixel_classify(pixel))

def generate_img():
    new_img = np.ones((350, 450, 3))
    for i in range (349):
        for j in range (349):
            pixel = (i, j)
            new_img[i][j] =  train_sample_rgb_mean[pixel_classify(pixel)]      
    new_img = np.array(new_img, dtype = np.uint8)   
    return new_img       
plt.imshow(generate_img())
plt.show()
