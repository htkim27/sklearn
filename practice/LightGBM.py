import pandas as pd
import numpy as np
from utils import utils as ut
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier


#import data
filename = '../data/iris_short.csv'
df = pd.read_csv(filename)
print(f'raw_data : {df.head()}')
print()
data = df.values
np.random.shuffle(data)

# Set variables
input_classes = ['versicolor', 'virginica']
test_size=0.3

# preprocessing
## split X y
X = data[:,2:-1]
y = data[:,-1]
## labelencoding
pre = ut.Preprocessing(X,y)
y = pre.cat_y_labeling(input_classes)
## train_test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

# Lightgbm
model = LGBMClassifier(objective = 'binary', n_estimators=100)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)

# validation
## accuracy_score
score = accuracy_score(y_test, y_preds)
print(f'accuracy_score : {score}')
print()

## confusion matrix
cm = confusion_matrix(y_test, y_preds)
print('confusion_matrix', cm, sep='\n')
print()

## classification_report
cr=classification_report(y_test, y_preds)
print('Classfication Report',cr,sep='\n')