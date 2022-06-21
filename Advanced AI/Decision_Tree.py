#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

path = os.path.realpath(os.curdir)
path = os.path.join(path, "banknote.xlsx")
# data = pd.read_excel("./Advanced AI/banknote.xlsx", sheet_name = "mini",
#                      header = 1, usecols="C:G")
data = pd.read_excel(path, sheet_name = "mini",
                     header = 1, usecols="C:G")
data = data.drop(["wav_krt", "pix_ent"], axis =1)
data.head()

features = data.drop(["class"], axis=1)
X = np.array(features)
y = np.array(data["class"])

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.75, 
                                              random_state=42)

ax = plt.subplot(111)
ax.set_facecolor((0.85, 0.85, 0.85))
ax.scatter(X_trn[:,0], X_trn[:,1], c=y_trn, edgecolors='k', alpha = 0.75, s =40)
ax.scatter(X_tst[:,0], X_tst[:,1], c=y_tst, edgecolors='w', alpha = 0.75, s =40)
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.show()
# %%
tree = DecisionTreeClassifier()
tree.fit(X_trn, y_trn)
y_trn_prd = tree.predict(X_trn)
print('Training accuracy:', accuracy_score(y_true=y_trn, 
                                           y_pred=y_trn_prd))
y_tst_prd = tree.predict(X_tst)
print('Testing accuracy:', accuracy_score(y_true=y_tst,
                                          y_pred=y_tst_prd))

x_min = X[:,0].min()-0.5; x_max = X[:, 0].max()+0.5
y_min = X[:,1].min()-0.5; y_max = X[:, 1].max()+0.5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.argmax(axis=1)
Z = Z.reshape(xx.shape)
ax = plt.subplot(111)
ax.contourf(xx, yy, Z, cmap="plasma",alpha = 0.25)
ax.scatter(X_trn[:,0],X_trn[:,1], c=y_trn,
           edgecolors="k", alpha=0.75)
ax.scatter(X_tst[:,0],X_tst[:,1], c=y_tst,
           edgecolors="w",alpha=0.75)
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])

dot_data = export_graphviz(tree, out_file=None,
                           feature_names=features.columns,
                           class_names=["REAL", "FAKE"],
                           filled=True,
                           rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph    
# %%
