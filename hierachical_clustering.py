import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# 변수 지정
filename = 'data/cluster_example_data.csv'
linkage_choice = 'ward'
n_clusters = 2

# data load
data = pd.read_csv(filename)
data = data.values #numpy array 객체로
np.random.shuffle(data) #data 배열 바꾸기

# data points 지정
X = data[:,:-1]
idx = data[:,-1]

# before clustering
for V in X:
    plt.scatter(V[0], V[1])
plt.title('Before Clustering')
plt.ylim(0,30)
plt.xlim(20,70)
plt.show()

# AgglomerativClustering
cluster = AgglomerativeClustering(linkage=linkage_choice, n_clusters=n_clusters)
cluster_idf = cluster.fit(X)
print(cluster_idf.labels_) #data index와 상관 없이 랜덤하게 정령되어 있음
result = pd.DataFrame(data=cluster_idf.labels_, # idx에 저장해둔 데이터 인덱스 넘버를 같이 print 해준다
                   index=idx,
                   columns=['cluster_num'])
result.rename_axis('data_num', inplace=True)
print(result)

# after clustering
result_val=result.values
for k in range(len(idx)):
    if result_val[k] == 0:
        plt.scatter(X[k][0],X[k][1], c='red')
    elif result_val[k] == 1:
        plt.scatter(X[k][0],X[k][1], c='blue')
    elif result_val[k] == 2:
        plt.scatter(X[k][0],X[k][1], c='green')
    elif result_val[k] == 3:
        plt.scatter(X[k][0],X[k][1], c='yellow')
plt.title('After Clustering')
plt.ylim(0, 30) # 경계 지정하기
plt.xlim(20, 70)
plt.show()

# silhouette_score -> 군집화 평가 및 적절한 클러스터 개수 탐구
min_cluster=2
max_cluster=10
sil_result=[]

# Dendrogram
np.set_printoptions(precision=5, suppress=True)
Z = linkage(X, 'ward')
plt.figure(figsize=(25,10))
plt.title('Hierarchical Clustering Dendogram', fontsize=18)
plt.xlabel('Observation ID', fontsize=18)
plt.ylabel('Distance', fontsize=18)
dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=14.,
)
plt.show()

# silhouette_score
for k in range(min_cluster,max_cluster+1):
    model=AgglomerativeClustering(linkage=linkage_choice, n_clusters=k)
    cluster=model.fit_predict(X)
    sil_score=silhouette_score(X, cluster)
    print(k, sil_score)
    sil_result.append(sil_score)
print('best score : ',np.argmax(sil_result)+min_cluster, sil_result[np.argmax(sil_result)])