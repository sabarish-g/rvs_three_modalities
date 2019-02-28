import numpy as np
import pdb
import os
import argparse
import scipy.io.wavfile as wav
from python_speech_features import mfcc

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

def main(args):
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
    
    if args.mode == 'train':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.trainImages.txt','r') as f:
            train_images = f.read().splitlines()
        print('the total train images are: %s' %len(train_images))
        train_images = sorted(train_images)
        #Getting only the training images and audio from the entire audio list
        index_list = [i for i,x in enumerate(image_list) if x in train_images]
        print('the total train images are: %s' %len(index_list))
        audio = []
        for val in index_list:
            audio.append(audio_list[val])
        print('the total audio files for training are: %s' %len(audio))

    elif args.mode == 'dev':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.devImages.txt','r') as f:
            dev_images = f.read().splitlines()
        print('the total dev images are: %s' %len(dev_images))
        dev_images = sorted(dev_images)
        #Getting only the dev images and audio from the entire audio list
        index_list = [i for i,x in enumerate(image_list) if x in dev_images]
        print('the total dev images are: %s' %len(index_list))
        audio = []
        for val in index_list:
            audio.append(audio_list[val])
        print('the total audio files for training are: %s' %len(audio))
    
    elif args.mode == 'test':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
            test_images = f.read().splitlines()
        print('the total test images are: %s' %len(test_images))   
        test_images= sorted(test_images)
        #Getting only the test images and audio from the entire audio list
        index_list = [i for i,x in enumerate(image_list) if x in test_images]
        print('the total test images are: %s' %len(index_list))
        audio = []
        for val in index_list:
            audio.append(audio_list[val])
        print('the total audio files for training are: %s' %len(audio))
    
    else:
        print("incorrect mode selected")
        
    audio_stack = np.array([])
    mean_stack = np.array([])
    for i in range(0,len(audio)):
        if i%100 == 0 and i!=0 : 
            print('extracted {}/{}'.format(i, len(audio)))
        af = audiofile_to_input_vector(os.path.join(args.data_path, audio[i]),29,29)
        mean_af = np.mean(af,axis=1)
        af2 = remap(af)
        af2 = np.reshape(af2,(1,2900))
        if len(audio_stack.shape)>1:
            audio_stack = np.vstack((audio_stack,af2))
        else:
            audio_stack = af2
    
    if args.mode == 'train':
        np.save(os.path.join(args.save_path, 'flickr8k_train_audio_features.npy'), audio_stack)
    if args.mode == 'dev':
        np.save(os.path.join(args.save_path, 'flickr8k_dev_audio_features.npy'), audio_stack)
    if args.mode == 'test':
        np.save(os.path.join(args.save_path, 'flickr8k_test_audio_features.npy'), audio_stack)
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='flickr', help='Data file')
    parser.add_argument('--save_path', type = str, default = '/shared/kgcoe-research/mil/new_cvs_data/audio_features', help = 'path to save the features')
    parser.add_argument('--data_path', type = str, default = '/shared/kgcoe-research/mil/new_cvs_data/flickr_audio/wavs', help = 'path to wav files')
    parser.add_argument('--mode', type=str, default='train', help='Feature extraction for which phase?')
    args=parser.parse_args()
    main(args)

