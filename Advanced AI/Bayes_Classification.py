#%%
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120


# %%

data_mini = pd.read_excel("banknote.xlsx", sheet_name = "mini", 
                     header= 1, usecols="C:G", nrows=101)

data_all = pd.read_excel("banknote.xlsx", sheet_name = "full", 
                     header= 1, usecols="C:G", nrows=1372)

# %%
## mini 2feauture
data_mini_2f = data_mini.drop(["wav_krt","pix_ent","class"], axis = 1)
X = np.array(data_mini_2f)
y = np.array(data_mini["class"])
print(X)
print(y)

# %%
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.5)

# %%
gnb = GaussianNB()
gnb.fit(X_trn, y_trn)

y_trn_prd = gnb.predict(X_trn)
print("Training accuracey:", accuracy_score(y_true=y_trn,
                                            y_pred=y_trn_prd))

y_tst_prd = gnb.predict(X_tst)
print("Testing accuracey:", accuracy_score(y_true=y_tst,
                                            y_pred=y_tst_prd))
# %%
c_dict = {
    0:0,
    1:1
}
clrs_y = np.zeros(y.shape)
for idx in range(len(y)):
    clrs_y[idx] = c_dict[y[idx]]

ax = plt.subplot(111)
plt.xlabel(data_mini_2f.columns[0])
plt.ylabel(data_mini_2f.columns[1])  

x_min = X[:,0].min()-0.5  
x_max = X[:,0].max()+0.5  
y_min = X[:,0].min()-0.5  
y_max = X[:,0].max()+0.5

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
    )  

Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.argmax(axis=1)
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, cmap="plasma", alpha = 0.5)
ax.scatter(X[:,0], X[:,1], c=clrs_y, cmap="plasma",
           alpha = 0.5, edgecolors = "k", s = 40) 
# %%
