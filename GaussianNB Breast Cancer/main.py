# import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

'''
dataset = load_breast_cancer()

for i in dataset:
    print("\n", i, "\n")
    print(dataset[i])

targets = dataset['target']
features = dataset['data']

train, test, train_targets, test_targets = train_test_split(features, targets, test_size = 0.45, random_state = 42)

print("train\n", train)
print("test\n", train)
print("train targets\n", train)
print("test targets\n", train)

gnb = GaussianNB()
model = gnb.fit(train, train_targets)
preds = gnb.predict(test)

print("test data\n", test_targets)
print("predicted data\n", preds)
print(accuracy_score(test_targets, preds)*100)
'''

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm 

np.set_printoptions(threshold=200)

datafile = np.loadtxt("data.txt", delimiter = ",", dtype=str)
mydata = datafile[1:, 1:].astype(float)
categoryText = datafile[1:, 0]
print("categories", categoryText)

categories = []
for i in categoryText:
    if i == "dog":
        categories.append(0)
    elif i == "Male":        
        categories.append(1)
    else:
        categories.append(2)

clf = svm.SVC(kernel="linear")
clf.fit(mydata, categories)

xmin = mydata[:,0].min() - 1
xmax = mydata[:,0].max() + 1
ymin = mydata[:,1].min() - 1
ymax = mydata[:,1].max() + 1

xlin = np.linspace(xmin, xmax, 100)
print("xlin\n", xlin)
ylin = np.linspace(ymin, ymax, 100)
print("ylin\n", ylin)
xmesh, ymesh = np.meshgrid(xlin, ylin)
print("xmesh\n", xmesh)
print("ymesh\n", ymesh)
rav =np.c_[xmesh.ravel(), ymesh.ravel()]
print("rav", rav)

zvals = clf.predict(rav)
print("zvals", zvals)
zvals = zvals.reshape(xmesh.shape)
print("zvals", zvals)

plt.contourf(xmesh, ymesh, zvals, cmap="gist_yarg")
plt.scatter(mydata[:,0],mydata[:,1],c = categories, cmap="spring", s=1)
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.show()