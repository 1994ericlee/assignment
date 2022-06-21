#%%
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


X, y = datasets.make_circles(n_samples=500,
                             factor=0.3,
                             noise=0.1)
ax = plt.subplot(111)
ax.set_facecolor((0.85, 0.85, 0.85))
ax.scatter(X[:,0], X[:,1], c=y, edgecolors='k', alpha=0.5)
ax.set_aspect('equal')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], np.sqrt(X[:, 0]**2 + X[:,1]**2),
           c = y, s =10)
ax.view_init(20, 35)
plt.show()
# %%
model = SVC(kernel='poly', degree=2, coef0=1, C=10)
model.fit(X, y)

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
xy_list = np.c_[xx.ravel(), yy.ravel()]
ax = plt.subplot(111)

Z = model.predict(xy_list)
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap='plasma', alpha=0.25)

Z = model.decision_function(xy_list)
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, colors='k', levels=[-1,0,1],
           linestyles=['--','-','--'], linewidths=1,
           alpha=0.5)

ax.scatter(X[:,0],X[:,1], c=y, edgecolors='k',alpha=0.5)
ax.set_aspect('equal')

plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=150, facecolors='none', edgecolors='gray',
    linewidths=0.5
)
# %%
