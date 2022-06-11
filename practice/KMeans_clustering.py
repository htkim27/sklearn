import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

filename= '../data/cluster_example_data.csv'
df=pd.read_csv(filename)

# set data points, data points
X = df.values
X=X[:,:-1]
##normalize
# X=normalize(X)

# Before Clustering
for V in X:
    plt.scatter(V[0], V[1])
plt.title('Before Clustering')
plt.ylim(0,30)
plt.xlim(20,70)
plt.show()

#KMeans
Num_Clusters=3
model=KMeans(n_clusters=Num_Clusters,max_iter=300)
clusters=model.fit_predict(X)
# print(clusters)

# After Clustering
cluster_nums = []
for cluster_num in clusters:
    cluster_nums.append(cluster_num)

for k in range(len(X)):
    if cluster_nums[k] == 0:
        plt.scatter(X[k][0],X[k][1], c='red')
    elif cluster_nums[k] == 1:
        plt.scatter(X[k][0],X[k][1], c='blue')
    elif cluster_nums[k] == 2:
        plt.scatter(X[k][0],X[k][1], c='green')
plt.ylim(0,30)
plt.xlim(20,70)
plt.show()

# silhouette_score -> 군집화 평가 및 적절한 클러스터 개수 탐구
min_cluster=2
max_cluster=20
sil_result=[]
for k in range(min_cluster,max_cluster+1):
    model=KMeans(n_clusters=k, max_iter=100)
    cluster=model.fit_predict(X)
    sil_score=silhouette_score(X, cluster)
    print(k, sil_score)
    sil_result.append(sil_score)
print('best score : ',np.argmax(sil_result)+min_cluster, sil_result[np.argmax(sil_result)])
