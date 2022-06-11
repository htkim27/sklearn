import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# neartest_distances among data points
def nearest_dist(data_points):
    norms=[]
    for i in range(len(data_points)):
        temp_norms = []
        for j in range(len(data_points)):
            if i != j:
                norm=np.linalg.norm(data_points[i]-data_points[j])
                temp_norms.append(norm)
        norms.append(min(temp_norms))
    return norms

# set variables

eps= 0.001
min_samples=2
# metric="euclidean"
metric="cosine"

filename='data/cluster_example_data.csv'

# create data
data=pd.read_csv(filename)
data=data.values

# set data points
X = data[:,:-1]
idx = data[:,-1]

# Before Clustering
for V in X:
    plt.scatter(V[0], V[1])
plt.title('Before Clustering')
plt.ylim(0,30)
plt.xlim(20,70)
plt.show()

# check the mean of the nearest distance to set epsilon (Euclidean)
nearest_dist_list = nearest_dist(X)
find_epsilon=np.array(nearest_dist_list).mean()
print(f'Epsilon can be {find_epsilon}')

#Standard Scaler (Choice)
# X_std=StandardScaler().fit_transform(X)

# clustering
cluster = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)
# cluster_std = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X_std)
print(f'DBSCAN result : {cluster}')
print()

# After Clustering
cluster_nums=[]
for doc_num, cluster_num in enumerate(cluster):
# for doc_num, cluster_num in enumerate(cluster_std):
    cluster_nums.append(cluster_num)
for k in range(len(idx)):
    if cluster_nums[k] == 0:
        plt.scatter(X[k][0],X[k][1], c='red')
    elif cluster_nums[k] == 1:
        plt.scatter(X[k][0],X[k][1], c='blue')
    elif cluster_nums[k] == 2:
        plt.scatter(X[k][0],X[k][1], c='green')
    elif cluster_nums[k] == 3:
        plt.scatter(X[k][0],X[k][1], c='yellow')
    elif cluster_nums[k] == 4:
        plt.scatter(X[k][0],X[k][1], c='orange')
    elif cluster_nums[k] == 5:
        plt.scatter(X[k][0],X[k][1], c='pink')
    else:
        plt.scatter(X[k][0],X[k][1], c='black')
plt.title('After Clustering')
plt.ylim(0, 30) # 경계 지정하기
plt.xlim(20, 70)
plt.show()

