import numpy as np
import pandas as pd
from utils import utils as ut
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# import data
filename='../data/iris_short.csv'
d = ut.Data()
df = d.import_df(filename)
print(d)
print()
data = df.values
 ## Set X, y
X = data[:,2:-1]
y = data[:,-1]

# categorical data preprocessing
input_classes = ['versicolor', 'virginica']
pre = ut.Preprocessing(X,y)
y = pre.cat_y_labeling(input_classes)

#train_test
 ## train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
 ## adaboost
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200)
model.fit(X_train, y_train)
print(f'AdaBoostClassifier test score : {model.score(X_test,y_test)}')
