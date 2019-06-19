import numpy as np
from numpy import genfromtxt
from  models import modeltrainer 

x_train  = genfromtxt('train_ml.csv', delimiter = ',')
x_test = genfromtxt('test_ml.csv', delimiter = ',')
y_train = genfromtxt('train_y.csv', delimiter = ',')
y_test = genfromtxt('test_y.csv', delimiter = ',')

clf = modeltrainer(x_train, x_test, y_train, y_test)
accuracy_score = clf.train_GradientBoost()
print(accuracy_score)

#print(clf.train_GradientBoost())
