import numpy as np
import pdb
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
A = np.load('/shared/kgcoe-research/mil/new_cvs_data/sentence_features/flickr8k_train_sentence_features.npy')
B = np.load('/shared/kgcoe-research/mil/new_cvs_data/audio_features/flickr8k_train_audio_features.npy')



    
out = []

pca = PCA(n_components = 1024)

A_PCA = pca.fit_transform(A)
# pdb.set_trace()
B_PCA = pca.fit_transform(B)
for i in range(0,A.shape[0]):
    out.append(cosine_similarity(A_PCA[i].reshape((1,A_PCA.shape[1])), B_PCA[i].reshape((1,B_PCA.shape[1]))))

pdb.set_trace()