#Maximum Likelihood Classification 

#%%
from random import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def getsamples(img):
    x, y, z = img.shape
    samples = np.empty([x * y, z])
    index = 0
    for i in range(x):
        for j in range(y):
            samples[index] = img[i, j]
            index += 1
    return samples


def EMSegmentation(img, no_of_clusters=5):
    output = img.copy()
    colors = np.array([[0, 11, 111], [22, 22, 22]])
    samples = getsamples(img)
    em = cv2.ml.EM_create()
    em.setClustersNumber(no_of_clusters)
    em.trainEM(samples)
    means = em.getMeans()
    covs = em.getCovs()  # Known bug: https://github.com/opencv/opencv/pull/4232
    x, y, z = img.shape
    distance = [0] * no_of_clusters
    for i in range(x):
        for j in range(y):
            for k in range(no_of_clusters):
                diff = img[i, j] - means[k]
                distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
            output[i][j] = colors[distance.index(max(distance))]
    return output


img = cv2.imread('irabu_zhang1.bmp')
output = EMSegmentation(img)
cv2.imshow('image', img)
cv2.imshow('EM', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
import cv2
import numpy as np

img = cv2.imread("irabu_zhang1.bmp")

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break

cv2.waitKey(0)
cv2.destroyAllWindow()
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
# print(classRange.shape)
# print(classRange)
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
    
# print(train_sample_pixel)
# print(train_sample_rgb)
#train_sample_rgb_mean = every class mean of rgb
train_sample_rgb_mean = np.mean(train_sample_rgb, axis = 1)
# print(train_sample_rgb_mean)

# r = train_sample_rgb[0][:, 0]
# g = train_sample_rgb[0][:, 1]
# b = train_sample_rgb[0][:, 2]

# sample = np.array([r, g, b])
# covMatrix = np.cov(sample, bias = True)
# inverseCovMatrix = np.linalg.inv(covMatrix)
# detCovMatrix = np.linalg.det (covMatrix)
# constant = -0.5*d*np.log(2*np.pi) -0.5*np.log(detCovMatrix)
# print(train_sample_rgb[0])
# print(train_sample_rgb_mean[0])

# s = test_sample[0] - train_sample_rgb_mean[0]
# prob_x = -0.5 * (np.dot(s, inverseCovMatrix)*s)
# print(prob_x)
# print(covMatrix)
pixel = (0,0)
def pixel_classify(pixel):
    pixel_rgb = np.asarray(pixel2rgb(pixel))
    
    classcode = cal_ml(pixel_rgb)
    return classcode

def pixel2rgb(pixel):
    pixel_rgb = img[pixel[1]][pixel[0]]
    # print(pixel_rgb)
    return pixel_rgb

def cal_ml(pixel_rgb):
    flag = 0
    old_prob = -1000000
    for i in range (class_number):
        # print(train_sample_rgb_mean[i])
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
    for i in range (449):
        for j in range (349):
            pixel = (i, j)
            new_img[:,:,:] =  train_sample_rgb_mean[pixel_classify(pixel)]      
    return new_img      
plt.imshow(generate_img())
plt.show()
# %%
