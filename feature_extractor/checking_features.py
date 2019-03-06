import numpy as np
from scipy.spatial.distance import cdist    
import pdb

sen_ft = np.load('/shared/kgcoe-research/mil/new_cvs_data/setnence_features/flickr8k_sentence_skipthoughts.npy')
aud_ft = np.load('/shared/kgcoe-research/mil/new_cvs_data/audio_features/flickr8k_audio_mfcc.npy')
aud_ft = np.float32(aud_ft)
# pdb.set_trace()

def cosine_vectorized(array1, array2):
    sumyy = (array2**2).sum(1)
    sumxx = (array1**2).sum(1, keepdims=1)
    sumxy = array1.dot(array2.T)
    return (sumxy/np.sqrt(sumxx))/np.sqrt(sumyy)
    

for i in range(0,100):
    arraysen = sen_ft[i].reshape((sen_ft[i].shape[0],1))
    arrayaud = aud_ft[i].reshape((aud_ft[i].shape[0],1))

    csim = cosine_vectorized(arraysen,arrayaud)
    print(np.sum(csim))