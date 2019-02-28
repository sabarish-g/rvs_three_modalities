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
    #Train,dev,test images list and sorting all lists to keep them in order
    if args.mode == 'train':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.trainImages.txt','r') as f:
            train_images = f.read().splitlines()
        print('the total train images are: %s' %len(train_images))
        sorted_images = sorted(train_images)
        image_features_np = np.zeros((len(sorted_images), 2048))

    elif args.mode == 'dev':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.devImages.txt','r') as f:
            dev_images = f.read().splitlines()
        print('the total dev images are: %s' %len(dev_images))
        sorted_images = sorted(dev_images)
        image_features_np = np.zeros((len(sorted_images), 2048))

    elif args.mode == 'test':
        with open('/shared/kgcoe-research/mil/new_cvs_data/Flickr8k_text/Flickr_8k.testImages.txt','r') as f:
            test_images = f.read().splitlines()
        print('the total test images are: %s' %len(test_images))
        sorted_images= sorted(test_images)
        image_features_np = np.zeros((len(sorted_images), 2048))
    
    else:
        print('incorrect mode selected')
    
    input_image = tf.placeholder(shape=[None, None, 3], dtype=tf.float32, name='input_image')
    preprocessed_image = vgg_preprocess(input_image, args.base_arch)
    features = feature_extractor(args, preprocessed_image, is_training=False)
    # Define the saver
    saver = tf.train.Saver()
       
    count=0
    image_to_feature={}

    with tf.Session() as sess:
        # Restore pre-trained weights
        saver.restore(sess, args.checkpoint)
        for i in range(0, len(sorted_images)):
            if i%10==0 and i!=0: print "Extracted: {}/{}".format(i, (len(sorted_images)))
            # sample is of form (image, caption)
            image = io.imread(os.path.join(args.images_path,sorted_images[i]))
            if len(image.shape)!=3: 
                image=gray2rgb(image)
            feature_val = sess.run(features, feed_dict={input_image: image})
            image_features_np[count] = np.squeeze(feature_val)
            count+=1
        # np.save(os.path.join(args.save_path, 'flickr_test_r152_precomp.npy'), image_features_np[0:len(image_data) - offset : 5, :])
        if args.mode == 'train':
            np.save(os.path.join(args.save_path, 'flickr8k_train_features.npy'), image_features_np)
        
        if args.mode == 'dev':
            np.save(os.path.join(args.save_path, 'flickr8k_dev_features.npy'), image_features_np)
        
        if args.mode == 'test':
            np.save(os.path.join(args.save_path, 'flickr8k_test_features.npy'), image_features_np)
        
        print "Total number of image features: {}".format(count)
        print "Done extracting Image features !!"
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/img_features/', help='Data file')
    parser.add_argument('--base_arch', type=str, default='resnet_v1_152', help='Base architecture of CNN')
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v1_152.ckpt', help='Path to checkpoint')
    parser.add_argument('--images_path', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/Flicker8k_Dataset', help='Path to images')
    parser.add_argument('--mode', type=str, default='train', help='Feature extraction for which phase?')
    args=parser.parse_args()
    main(args)

