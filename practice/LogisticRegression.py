import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# set variables
filename= '../data/iris_data.csv'
label_classes=['versicolor','virginica']
lr_C=0.5
lr_penalty='l2'
lr_solver='liblinear'
lr_max_iter=1000


# import data
data=pd.read_csv(filename)
data_np=data.values
np.random.shuffle(data_np)

# split data into labels and features
data_labels=data_np[:,-1]
data_features=data_np[:,:-1]

# transfrom labels into categorical variables(num)
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(label_classes)
data_labels=labelEncoder.transform(data_labels)

# train, test split
X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.3)

# HyperParmeterTuning
# GridSearch
target_model = LogisticRegression(max_iter=lr_max_iter)
params = {'penalty':['l1','l2'],
          'C':[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
          'solver':['liblinear']}
grid_search=GridSearchCV(target_model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)
print(f'result of grid_search:{grid_search.best_params_}')

best_model = grid_search.best_estimator_
y_preds = best_model.predict(X_test)
print(f"best_model's accuracy:{accuracy_score(y_test, y_preds)}")
print()

# Logistic Regression
# train
model = LogisticRegression(C=lr_C, penalty=lr_penalty, solver=lr_solver, max_iter=lr_max_iter)
model.fit(X_train, y_train)
print("model's coef : ",model.coef_)
print("model's intercept : ",model.intercept_)
# test
y_prediction=model.predict(X_test)
print("results of prediction : ",y_prediction)
print("labels of predicted values : ",y_test)
print("model's test score : ",model.score(X_test,y_test))
print()

# Confusion Matrix
confusionMatrix = confusion_matrix(y_test, y_prediction)
print('Confusion Matrix',confusionMatrix,sep='\n')
print()

# Classification Report
classificationReport=classification_report(y_test, y_prediction)
print('Classfication Report',classificationReport,sep='\n')

y_prob = model.predict_proba(X_test)
# ROC_AUC score
roc_aucScore = roc_auc_score(y_test,y_prob[:,1])
print('ROC_AUC SCORE : ',roc_aucScore)

# Printing ROC CURVE
fpr, tpr, thresh = roc_curve(y_test, y_prob[:,1], pos_label=1)

#for printing tpr=fpr
random_prob = [0 for i in range(len(y_test))]
p_fpr, p_tpr, p_tresh = roc_curve(y_test, random_prob, pos_label=1)

plt.style.use('seaborn')

plt.plot(fpr,tpr,linestyle='--',color='orange',label='Logistic Regression')
plt.plot(p_fpr,p_tpr,linestyle='--',color='blue')
plt.title('ROC curve')
plt.xlabel('FPR (FalsePositiveRate)')
plt.ylabel('TPR (TruePositiveRate)')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()