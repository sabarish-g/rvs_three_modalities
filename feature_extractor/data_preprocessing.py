#Making the train, test files from given txt file.
import tensorflow as tf
import numpy as np
import argparse
from dnn_library import *
from skimage import io
from skimage.transform import resize
from preprocessing import preprocessing_factory
from nets import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import cv2
import pdb
from skimage.color import gray2rgb

#Train,dev,test images list
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

'''
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
index_list = [i for i,x in enumerate(image_list) if x in train_images]
print('the total train images are: %s' %len(index_list))

train_audio = []
for val in index_list:
    train_audio.append(audio_list[val])
print('the total audio files for training are: %s' %len(train_audio))

#Dealing with the captions
with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr8k.lemma.token.txt','r') as f:
    captions = f.read().splitlines()
file_list=[]
captions_list=[]
for line in captions:
    file_list.append(line.split('#')[0])
    sentence = line.split('#')[1].split('\t')[1]
    captions_list.append(sentence)

train_sentences = []
cap_files = []
for element in train_images:
    idx_list = [i for i,val in enumerate(file_list) if val==element]
    for i in idx_list:
        train_sentences.append(captions_list[i])
        cap_files.append(file_list[i])

print('the total captions files for training are: %s' %len(train_sentences))
'''

def feature_extractor(args, image, reuse=None, is_training=False):
    """
    Builds the model architecture
    """			
    # Define the network and pass the input image
    with slim.arg_scope(model[args.base_arch]['scope']):
            logits, end_points = model[args.base_arch]['net'](image, num_classes=1000, is_training=False) #model[args.base_arch]['num_classes']
    

    # features 
    feat_anchor = tf.squeeze(end_points[model[args.base_arch]['end_point']])
        
    return feat_anchor
    

def feature_extractor(args, image, reuse=None, is_training=False):
    """
    Builds the model architecture
    """			
    # Define the network and pass the input image
    with slim.arg_scope(model[args.base_arch]['scope']):
            logits, end_points = model[args.base_arch]['net'](image, num_classes=1000, is_training=False) #model[args.base_arch]['num_classes']
    

    # features 
    feat_anchor = tf.squeeze(end_points[model[args.base_arch]['end_point']])
        
    return feat_anchor

def vgg_preprocess(image, base_arch):
    
    """
    Pre-processing for base network.
    """
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(base_arch, is_training=False)
    return tf.expand_dims(image_preprocessing_fn(image, 224, 224), 0)

def main(args):
    input_image = tf.placeholder(shape=[None, None, 3], dtype=tf.float32, name='input_image')
    preprocessed_image = vgg_preprocess(input_image, args.base_arch)
    features = feature_extractor(args, preprocessed_image, is_training=False)
    # Define the saver
    saver = tf.train.Saver()
    print "Total number of samples: {}".format(len(train_images))

    # Define the TF record writer
    tfrecord_writer = tf.python_io.TFRecordWriter(args.record_path)
    image_features_np = np.zeros((len(train_images), 2048))
    # image_features_np = np.zeros((1000, 2048))
    count=0
    image_to_feature={}

    with tf.Session() as sess:
        # Restore pre-trained weights
        saver.restore(sess, args.checkpoint)
        for i in range(0, len(train_images)):
            if i%10==0 and i!=0: print "Extracted: {}/{}".format(i, (len(train_images)))
            # sample is of form (image, caption)
            image = io.imread(os.path.join(args.images_path,train_images[i]))
            if len(image.shape)!=3: 
                image=gray2rgb(image)
            # pdb.set_trace()
            # augment image and mean subtraction
             
            # images = augment_image(image) 
            # mean_subtracted_image = images - np.array([_R_MEAN, _G_MEAN, _B_MEAN])
            # Run  the session to extract features
            # pdb.set_trace()
            feature_val = sess.run(features, feed_dict={input_image: image})
            # pdb.set_trace()
            # mean_img_features = np.mean(feature_val, axis=0)
            image_features_np[count] = np.squeeze(feature_val)
            count+=1
        # np.save(os.path.join(args.save_path, 'flickr_test_r152_precomp.npy'), image_features_np[0:len(image_data) - offset : 5, :])
        np.save(os.path.join(args.save_path, 'flickr8k_train_r152_precomp.npy'), image_features_np)
        # tfrecord_writer.close()
        print "Total number of image features: {}".format(count)
        print "Done extracting Image features !!"
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='flickr', help='Data file')
    # parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/coco/train_filenames.txt', help='Data file')
    # parser.add_argument('--caps_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/data/coco_precomp/train_caps.txt', help='Data file')
    parser.add_argument('--save_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/img_features', help='Data file')
    # parser.add_argument('--root_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/mscoco_skipthoughts/images', help='Root_path')
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/img_features/train_images.tfrecord', help='Root_path')
    parser.add_argument('--base_arch', type=str, default='resnet_v1_152', help='Base architecture of CNN')
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v1_152.ckpt', help='Path to checkpoint')
    parser.add_argument('--images_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/Flicker8k_Dataset', help='Path to images')
    args=parser.parse_args()
    main(args)

