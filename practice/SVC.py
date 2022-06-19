import numpy as np
import pandas as pd
from utils import utils as ut
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data import
filename = '../data/iris_short.csv'
df = pd.read_csv(filename)
data = df.values
print('raw_data', df.head(), sep='\n')
print()

#set variables
input_classes = ['virginica', 'versicolor']
test_size = 0.3
C=1
kernel='rbf'
gamma='auto'

# X y split
X = data[:,:-1]
y = data[:,-1]
# labeling categorical data
pre=ut.Preprocessing(X,y)
y = pre.cat_y_labeling(input_classes)
# train_test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

# SVC(Support Vector Classifier)
# train
model = SVC(C=C, kernel=kernel, gamma=gamma)
model.fit(X_train, y_train)

#test
y_preds = model.predict(X_test)
score = accuracy_score(y_test, y_preds)
cm = confusion_matrix(y_test, y_preds)
cr = classification_report(y_test, y_preds)
print(f'SVC_accuracy_score : {score}')
print('SVC Confusion Matrix', cm, sep='\n')
print('SVC Classification Report', cr, sep='\n')