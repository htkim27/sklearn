import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Corpus
corpus = ['banana apple apple eggplant',
        'orange carrot banana eggplant',
        'apple carrot banana banana',
        'orange banana grape']

# hyperparameter
CV_min_df=1
CV_ngram_range=(1,2)

# CountVectorizer
vectorizer=CountVectorizer(min_df=CV_min_df, ngram_range=CV_ngram_range)
features=vectorizer.fit_transform(corpus)

# tf_features
dtm = np.array(features.todense())
print(f'Doc_Term Matrix in numpy :\n{dtm}\n')

# 벡터 간 거리 norm, cosine
comp_idx=[0,1]
d1=dtm[comp_idx[0]]
d2=dtm[comp_idx[1]]
l2_norm = np.linalg.norm(d1-d2)
cos_sim = np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))
print(f'l2norm between doc{comp_idx[0]} and doc{comp_idx[1]} : {l2_norm}\n')
print(f'cosine_similarity between doc{comp_idx[0]} and doc{comp_idx[1]} : {cos_sim}\n')

# doc_term Matrix 시각화
terms = vectorizer.get_feature_names_out()
visualized_dtm = pd.DataFrame(data=dtm,
                              columns=terms,
                              index=['DOC'+str(i) for i in range(len(dtm))])
print('Visualized Document_Term Matrix', visualized_dtm, sep='\n')