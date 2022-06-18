import pandas as pd
import numpy as np
from utils import utils as uts
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import data
filename = '../data/iris_short.csv'
df = pd.read_csv(filename)
data = df.values

# set variables
input_classes = ['versicolor','virginica']
test_size = 0.3
max_depth=2
n_estimators=1000

X = data[:,:-1]
y = data[:,-1]

# categorical labeling
pre = uts.Preprocessing(X,y)
y = pre.cat_y_labeling(input_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)
model.fit(X_train, y_train)
model_score = model.score(X_test, y_test)
print(f'XGBoost score : {model_score}')