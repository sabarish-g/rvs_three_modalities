import numpy as np
import pdb
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import skipthoughts


'''
Dev images + audio+ captions
'''

with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
    test_images = f.read().splitlines()
print('the total test images are: %s' %len(test_images))   
#sorting all lists to keep them in order

test_images= sorted(test_images)


#getting all the wave files
with open('/shared/kgcoe-research/mil/new_cvs_data/flickr_audio/wav2capt.txt','r') as f:
    names = f.read().splitlines()    
audio_list = []
image_list = []
for index in names:
    audio_list.append(index.split(' ')[0])
    image_list.append(index.split(' ')[1]) 
print('the total audio files are: %s' %len(audio_list))    
print('the total images are: %s' %len(image_list))    

audio_list= sorted(audio_list)
image_list= sorted(image_list)

#Getting only the training images and audio from the entire audio list
index_list = [i for i,x in enumerate(image_list) if x in test_images]
print('the total test images are: %s' %len(index_list))

test_audio = []
for val in index_list:
    test_audio.append(audio_list[val])
print('the total audio files for test are: %s' %len(test_audio))



#test audio features

def remap(arr):
    old_size = arr.shape[0]#100
    new_shape = 100#arr.shape[0]
    new_arr = np.zeros((new_shape,arr.shape[1]))
    new_idxs = [i*old_size//new_shape + old_size//(2*new_shape) for i in range(new_shape)]
    for n,idx in enumerate(new_idxs):
        new_arr[n,:] = arr[idx,:]
    return new_arr
    
    
def audioToInputVector(audio, fs, numcep, nfilt):
    # Get MFCC coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep, nfilt=nfilt)
    return features


def audiofile_to_input_vector(audio_filename, numcep, nfilt):
    '''
    Given a WAV audio file at `audio_filename`, calculates `numcep` MFCC features
    at every time step.
    '''
    # Load .wav file
    fs, audio = wav.read(audio_filename)
    #pdb.set_trace()
    return audioToInputVector(np.int16(audio), fs, numcep, nfilt)


audio_stack = np.array([])
mean_stack = np.array([])
test_save_audio_path = '/shared/kgcoe-research/mil/new_cvs_data/audio_features/test'
test_audio_path = '/shared/kgcoe-research/mil/new_cvs_data/flickr_audio/wavs'
for i in range(0,len(test_audio)):
    if i%100 == 0 and i!=0 : 
        print('extracted {}/{}'.format(i, len(test_audio)))
    af = audiofile_to_input_vector(os.path.join(test_audio_path, test_audio[i]),29,29)
    mean_af = np.mean(af,axis=1)
    #pdb.set_trace()
    af2 = remap(af)
    af2 = np.reshape(af2,(1,2900))
    # pdb.set_trace()
    if len(audio_stack.shape)>1:
        # audio_stack = np.vstack((audio_stack,af2[...,np.newaxis]))
        audio_stack = np.vstack((audio_stack,af2))
        # mean_stack  = np.vstack((mean_stack,mean_af[...,np.newaxis]))
    else:
        audio_stack = af2
        mean_stack = mean_af

pdb.set_trace()
np.save(test_save_audio_path, audio_stack)