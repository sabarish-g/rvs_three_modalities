from __future__ import absolute_import, print_function
import argparse
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import os
import pdb


#Train,dev,test images list
#Putting all the file names into respective lists
with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.trainImages.txt','r') as f:
    train_images = f.read().splitlines()
print('the total train images are: %s' %len(train_images))

with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.devImages.txt','r') as f:
    dev_images = f.read().splitlines()
print('the total dev images are: %s' %len(dev_images))

with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
    test_images = f.read().splitlines()
print('the total test images are: %s' %len(test_images))   
#sorting all lists to keep them in order
train_images = sorted(train_images)
dev_images = sorted(dev_images)
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


#sorting all lists to keep them in order
audio_list= sorted(audio_list)
image_list= sorted(image_list)

#Getting only the training images and audio from the entire audio list
index_list = [i for i,x in enumerate(image_list) if x in train_images]
print('the total train images are: %s' %len(index_list))

#Getting only the training wavs.
train_audio = []
for val in index_list:
    train_audio.append(audio_list[val])
print('the total audio files for training are: %s' %len(train_audio))


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


#audio_features = audiofile_to_input_vector('/shared/kgcoe-research/mil/Kruthika/AllAugmentedSenecaData25/SJ_DeepSpeech_SJ_deer_habits_23.wav', 29, 29)
#pdb.set_trace()

def main(args):   
    audio_features_np = np.array([])
    for i in range(0, len(train_audio)):
        if i%500==0 and i!=0: print("Extracted: {}/{}".format(i, (len(train_audio))))
        audio_file = os.path.join(args.data_path, train_audio[i])
        pdb.set_trace()
        audio_features_np = np.append(audio_features_np,audiofile_to_input_vector(audio_file,29,29))
        
    np.save(os.path.join(args.save_path, 'flickr8k_audio_mfcc.npy'), audio_features_np)
        

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='flickr', help='Data file')
    parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/flickr_audio/wavs', help='Data file')
    parser.add_argument('--save_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/audio_features', help='Data file')
    args=parser.parse_args()
    main(args)

