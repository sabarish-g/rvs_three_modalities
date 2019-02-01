import numpy as np
import pdb
import os

print('Loading the features')
image_features = np.load('/shared/kgcoe-research/mil/new_cvs_data/img_features/flickr8k_train_r152_precomp.npy')
image_features = np.repeat(image_features, repeats = 5, axis = 0)
text_features = np.load('/shared/kgcoe-research/mil/new_cvs_data/setnence_features/flickr8k_sentence_skipthoughts.npy')
audio_features = np.load('/shared/kgcoe-research/mil/new_cvs_data/audio_features/flickr8k_audio_mfcc.npy')

print('The shape of image featues is %s' %str(image_features.shape))
print('The shape of text featues is %s' %str(text_features.shape))
print('The shape of audio featues is %s' %str(audio_features.shape))

image_path = '/shared/kgcoe-research/mil/new_cvs_data/img_features/splitted'
text_path = '/shared/kgcoe-research/mil/new_cvs_data/setnence_features/splitted'
audio_path = '/shared/kgcoe-research/mil/new_cvs_data/audio_features/splitted'

for i in range(0,image_features.shape[0]):
    if i%100 == 0 and i > 0: print('Split :{}/{}'.format(i, image_features.shape[0]))
    img_ft = image_features[i,]
    img_ft = np.reshape(img_ft,(2048,1))
    img_ft = np.transpose(img_ft)
    np.save(os.path.join(image_path, "img_"+str(i)+'.npy'), img_ft)
print('splitting of image features is done!')

print('Audio and text feature extraction')
for i in range(0,text_features.shape[0]):
    if i%100 == 0 and i > 0: print('Split :{}/{}'.format(i, text_features.shape[0]))
    txt_ft = text_features[i,]
    txt_ft = np.reshape(txt_ft,(4800,1))
    txt_ft = np.transpose(txt_ft)
    np.save(os.path.join(text_path, "txt_"+str(i)+'.npy'), txt_ft)
    # pdb.set_trace()
print('splitting of text features is done!')
    
for i in range(0,audio_features.shape[0]):
    if i%100 == 0 and i > 0: print('Split :{}/{}'.format(i, audio_features.shape[0]))
    aud_ft = audio_features[i,]
    aud_ft = np.reshape(aud_ft,(2900,1))
    aud_ft = np.transpose(aud_ft)
    np.save(os.path.join(audio_path, "aud_"+str(i)+'.npy'), aud_ft)
    # pdb.set_trace()
print('splitting of audio features is done!')
    