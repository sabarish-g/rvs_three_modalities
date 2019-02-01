import tensorflow as tf
import numpy as np
import os
from preprocessing import preprocessing_factory
import argparse
import pdb

class FlickrDataLoader(object):
    """
    Data loader and writer object for MSCOCO dataset
    """
    def __init__(self, path=None):
    # def __init__(self, path=None, precompute=False, use_random_crop=False, max_len=None, model='vse'):
        self.data_path=path
        # self.precompute=precompute
        # self.use_random_crop=use_random_crop
        # self.max_len=max_len
        # self.model=model

    def _bytes_feature(self, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

    def _bytes_feature_list(self, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
        return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])


        
    # def _make_single_example(self, image_path, caption, precompute=False):
    def _make_single_example(self, img_npy, cap_npy, precompute=False):
        """
        Make a single example in a TF record
        """
        # if not precompute:
            # image = tf.gfile.FastGFile(image_path, "rb").read()
            # caption=self._process_caption(caption)
            # caption_list = caption.split(' ')
            # feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
                       
            # context = tf.train.Features(feature={
                            # "image": self._bytes_feature(image)})
                            
        # else:
            # caption=self._process_caption(caption).strip()
            # caption_list = caption.split(' ')
            # feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
            # context = tf.train.Features(feature={"image": self._bytes_feature(image_path.tostring())})
        

        feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
        context = tf.train.Features(feature={"image": self._bytes_feature(image_path.tostring())})
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        return sequence_example
        
        
        
    def _precomputed_dataset(self, phase, record_path, image_ft, text_ft, num=None):
    """
    Write the whole dataset to a TF record.
    '/shared/kgcoe-research/mil/new_cvs_data/img_features/flickr8k_train_r152_precomp.npy'
    '/shared/kgcoe-research/mil/new_cvs_data/setnence_features/flickr8k_sentence_skipthoughts.npy'
    """
    
    image_embeddings = np.load(image_ft)
    train_img_features = np.repeat(image_embeddings, repeats = 5, axis = 0)
    train_text_features = np.load(text_ft)
    
    train_img_features = np.load(args.feature_path).astype(np.float32)
    tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
    # if num is None:
        # num=len(train_img_features)
    # count=0
    # 
    # for im_idx in range(num):
        # if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)                
        # for cap_idx in range(im_idx*5, im_idx*5 +5):
            # example = self._make_single_example(train_img_features[im_idx], train_caps[cap_idx].strip(), precompute=True)
            # tfrecord_writer.write(example.SerializeToString())
            # count+=1
    # pdb.set_trace()        
    # for im_idx in range(0, num, 5):
        # if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)                
        # for cap_idx in range(im_idx, im_idx +5):
            # example = self._make_single_example(train_img_features[im_idx], train_text_features[cap_idx].strip(), precompute=True)
            # tfrecord_writer.write(example.SerializeToString())
            # count+=1
    for i in range(0, train_img_features.shape[0]):
        if count%5000==0 and count!=0: print "Generated: {}/{}".format(i, train_img_features.shape[0])   
        example = self._make_single_example(train_img_features[i], train_text_features[i], precompute=True)
        tfrecord_writer.write(example.SerializeToString())
        
    print "Done generating TF records"    