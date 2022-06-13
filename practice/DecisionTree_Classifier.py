import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#grid search
def grid_search_clf(params, clf):
    # Instantiate GridSearchCV as grid_reg
    grid_clf = GridSearchCV(clf, params, cv=5, n_jobs=-1)

    # Fit grid_reg on X_train and y_train
    grid_clf.fit(X_train, y_train)
    best_params = grid_clf.best_params_
    print(f'best_params : {best_params}')

    # Extract best estimator
    best_model = grid_clf.best_estimator_

    # Extract best score
    best_score = grid_clf.best_score_

    # Print best score
    print("Training score: {:.3f}".format(best_score))

    # Predict test set labels
    y_pred = best_model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print accuracy
    print('Test score: {:.3f}'.format(accuracy))

    # Return best model
    return best_model

# import data
df = pd.read_csv('../data/heart_disease.csv')

# set variables
test_size = 0.2
grid_search_clf_params = {'criterion':['entropy', 'gini'],
                          'min_samples_split':[2, 3, 4, 5, 6, 8, 10],
                          'min_samples_leaf':[0.01, 0.02, 0.03, 0.04, 0.05],
                          'max_leaf_nodes':[10, 15, 20, 25, 30, 35, 40, 45, 50],
                          'max_depth':[2,4,6,8]
                          }

# Set X, y
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# train
 ## cross validation
model = DecisionTreeClassifier()
scores = cross_val_score(model,X,y,cv=5)
print(f'Cross Validation Scores : {np.round(scores,2)}')
print(f'Mean of CVS : {scores.mean()}')
print()

 ## grid search
best_model = grid_search_clf(params=grid_search_clf_params,
                             clf=DecisionTreeClassifier())
print(f'best model by grid searching : {best_model}')

  ### fit with results of grid search
model = DecisionTreeClassifier(criterion=best_model.criterion,max_depth=best_model.max_depth,
                               max_leaf_nodes=best_model.max_leaf_nodes, min_samples_leaf=best_model.min_samples_leaf,
                               min_samples_split=best_model.min_samples_split)
model.fit(X,y)
print(f"best_model's score : {model.score(X,y)}")
print()

# information of features
 ## importance of featurs
print(f"feature_importances' array : {model.feature_importances_}")
print(f"feature_importances' sum : {np.sum(model.feature_importances_)}")

 ## feature name visualizing
feature_dict = dict(zip(X.columns, model.feature_importances_))
feature_importances = [(k,v) for k, v in sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)]
print(f"feature_importance visualizing : {feature_importances}")