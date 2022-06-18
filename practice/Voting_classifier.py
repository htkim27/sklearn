import numpy as np
import pandas as pd
from utils import utils as ut
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# import data
filename='../data/iris_short.csv'
d = ut.Data()
df = d.import_df(filename)
data = d.df_to_data()
np.random.shuffle(data)

#set variables
input_classes = ['virginica', 'versicolor']
test_size = 0.3

# X y split
X = data[:,:-1]
y = data[:,-1]
# labeling categorical data
pre=ut.Preprocessing(X,y)
y = pre.cat_y_labeling(input_classes)
# train_test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

# Voting classifier
## set candidates
m1 = LogisticRegression(multi_class='multinomial')
m2 = RandomForestClassifier(n_estimators=50)
m3 = GaussianNB()

# voting=hard
em = VotingClassifier(estimators=[('lr',m1),('rf',m2),('gnp',m3)], voting='hard')
em.fit(X_train, y_train)
y_preds = em.predict(X_test)
score = accuracy_score(y_preds, y_test)
cm = confusion_matrix(y_test, y_preds)
cr = classification_report(y_test, y_preds)
print('=====VotingClassifier(hardvoting)=====')
print(f'accuracy score : {score}')
print('confusionmatix',cm,sep='\n')
print('classification_report', cr, sep='\n')

# voting=soft
em = VotingClassifier(estimators=[('lr',m1),('rf',m2),('gnp',m3)], voting='soft')
em.fit(X_train, y_train)
y_preds = em.predict(X_test)
score = accuracy_score(y_preds, y_test)
cm = confusion_matrix(y_test, y_preds)
cr = classification_report(y_test, y_preds)
print('=====VotingClassifier(softvoting)=====')
print(f'accuracy score : {score}')
print('confusionmatix',cm,sep='\n')
print('classification_report', cr, sep='\n')

# voting=soft, weight
em = VotingClassifier(estimators=[('lr',m1),('rf',m2),('gnp',m3)], voting='soft', weights=[1,2,1])
em.fit(X_train, y_train)
y_preds = em.predict(X_test)
score = accuracy_score(y_preds, y_test)
cm = confusion_matrix(y_test, y_preds)
cr = classification_report(y_test, y_preds)
print('=====VotingClassifier(softvoting & weights)=====')
print(f'accuracy score : {score}')
print('confusionmatix',cm,sep='\n')
print('classification_report', cr, sep='\n')