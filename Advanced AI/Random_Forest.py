#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

X, y = make_moons(n_samples = 1000, noise=0.25, random_state=1)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                              test_size=0.25,
                                              random_state=42)
ax = plt.subplot(111)
ax.set_facecolor((0.85, 0.85, 0.85))
ax.scatter(X_trn[:,0], X_trn[:,1], c=y_trn, edgecolors='k', alpha = 0.75, s =40)
ax.scatter(X_tst[:,0], X_tst[:,1], c=y_tst, edgecolors='w', alpha = 0.75, s =40)
plt.show()
# %%
forest = RandomForestClassifier(n_estimators=500, max_depth=7)
forest.fit(X_trn, y_trn)

y_trn_prd = forest.predict(X_trn)
print('Trainning accuracy:', accuracy_score(y_true = y_trn,
                                            y_pred = y_trn_prd))
y_tst_prd = forest.predict(X_tst)
print('Trainning accuracy:', accuracy_score(y_true = y_tst,
                                            y_pred = y_tst_prd))
# %%
x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)
Z = forest.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.argmax(axis=1)
Z = Z.reshape(xx.shape)
ax = plt.subplot(111)
ax.contourf(xx, yy, Z, cmap = "plasma", alpha = 0.25)
ax.scatter(X_trn[:,0],X_trn[:,1], c=y_trn,
           edgecolors="k", alpha=0.15)
ax.scatter(X_tst[:,0],X_tst[:,1], c=y_tst,
           edgecolors="w",alpha=0.15)

plt.show()
# %%
