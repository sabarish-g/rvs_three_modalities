import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import argparse
from model import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from preprocessing import preprocessing_factory
# from data.coco_data_loader import *
import pdb
import time



def order_sim_gpu(images_placeholder, text_placeholder):
    """
    Computes the order similarity between images and captions
    """
    clip_diff = tf.maximum(tf.subtract(text_placeholder, images_placeholder), 0)    
    sqr_clip_diff = tf.square(clip_diff)
    sim = tf.sqrt(tf.reduce_sum(sqr_clip_diff, axis=-1))
    sim = -tf.transpose(sim)
    
    return sim 

def t2i_gpu(image_embeddings, text_embeddings, measure='order', shard_size=25):
    """
    Text-Image retrieval on GPU (much faster compared to CPU impl. Refer to legacy for cpu imp)
    Args: 
        image_embeddings: 5000 x emb_dim
        text_embeddings: 5000 x emb_dim
    Returns: 
         Recall scores and ranks
    """
    # Runs a batch of 50 text samples with all other image embeddings in the dataset
    # Tiling to replicate each text sample to match number of total image samples
    text_tensor = tf.placeholder(shape=(shard_size, image_embeddings.shape[1]), dtype=tf.float32)
    image_tensor = tf.placeholder(shape=(image_embeddings.shape[0]/5, image_embeddings.shape[1]), dtype=tf.float32)

    text_exp_tensor = tf.expand_dims(text_tensor, 1)
    tile_text_embeddings =  tf.tile(text_exp_tensor, [1, image_embeddings.shape[0]/5, 1], name='tile_text_embeddings')

    image_exp_tensor = tf.expand_dims(image_tensor, 0)  
    tile_image_embeddings = tf.tile(image_exp_tensor, [shard_size, 1, 1], name='tile_image_embeddings')
    
    if measure=='order':
        d = order_sim_gpu(tile_image_embeddings,tile_text_embeddings)
        inds = tf.contrib.framework.argsort(d,direction="DESCENDING",axis=0)
    
    inds_np=np.zeros((image_embeddings.shape[0], image_embeddings.shape[0]/5), dtype=np.int32)
    # Unique image embeddings in the 5000 replicated original image_embeddings
    unique_im_embeddings = image_embeddings[0:image_embeddings.shape[0]:5]

    if measure=='order':
        with tf.Session() as sess:
            for i in range(0, inds_np.shape[0], shard_size):
                idx = sess.run(inds, feed_dict={image_tensor:unique_im_embeddings,
                                          text_tensor: text_embeddings[i:i+shard_size]})
                inds_np[i: i+shard_size, :] = idx.T

    elif measure=='cosine':
        sim_scores = np.matmul(text_embeddings, unique_im_embeddings.T)
        inds_np = np.argsort(sim_scores)[:, ::-1]
        
    ranks = np.zeros(inds_np.shape[0])
    top1 = np.zeros(inds_np.shape[0])

    for i in range(0, image_embeddings.shape[0]/5):
        for index in range(5):
            ranks[5 * i + index] = np.where(inds_np[5 * i + index] == i)[0][0]
            top1[5 * i + index] = inds_np[5 * i + index][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # R@1
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)  # R@5
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) # R@10

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1), inds_np

def i2t_gpu(image_embeddings, text_embeddings, measure='order', shard_size=25):
    """
    Image-Text retrieval on GPU (much faster compared to CPU impl. Refer to legacy for cpu imp)
    Args: 
        image_embeddings: 5000 x emb_dim
        text_embeddings: 5000 x emb_dim
    Returns: 
         Recall scores and ranks
    """
    # Runs a batch of 50 image samples with all other text embeddings in the dataset
    # Tiling to replicate each image sample to match number of total image samples
    text_tensor = tf.placeholder(shape=(text_embeddings.shape[0], image_embeddings.shape[1]), dtype=tf.float32)
    image_tensor = tf.placeholder(shape=(shard_size, image_embeddings.shape[1]), dtype=tf.float32)
    
    text_exp_tensor = tf.expand_dims(text_tensor, 0)
    tile_text_embeddings =  tf.tile(text_exp_tensor, [shard_size, 1, 1], name='tile_text_embeddings')
    
    image_exp_tensor = tf.expand_dims(image_tensor, 1)  
    tile_image_embeddings = tf.tile(image_exp_tensor, [1, text_embeddings.shape[0], 1], name='tile_image_embeddings')
    
    if measure=='order':
        d = order_sim_gpu(tile_image_embeddings, tile_text_embeddings)
        inds = tf.contrib.framework.argsort(d,direction="DESCENDING",axis=0)

    unique_im_emb = image_embeddings[0:text_embeddings.shape[0]:5,:]
    inds_np=np.zeros((unique_im_emb.shape[0], text_embeddings.shape[0]), dtype=np.int32)
    
    if measure=='order':
        with tf.Session() as sess:
            for i in range(0, unique_im_emb.shape[0], shard_size):
                idx = sess.run(inds, feed_dict={image_tensor:unique_im_emb[i: i+shard_size],
                                          text_tensor: text_embeddings})
                inds_np[i: i+shard_size, :] = idx.T
            
    elif measure=='cosine':
        sim_scores = np.matmul(unique_im_emb, text_embeddings.T)
        inds_np = np.argsort(sim_scores)[:, ::-1]
    
    ranks = np.zeros(unique_im_emb.shape[0], dtype=np.int32)
    top1 = np.zeros(unique_im_emb.shape[0], dtype=np.int32)
    
    for i in range(inds_np.shape[0]):
        rank = 1e20
        for index in range(5*i, 5*i + 5, 1):
            tmp = np.where(inds_np[i] == index)[0][0]   # Actual GT indices are 10*index to 10*index +5. tmp is the rank of these items.
            if tmp < rank:
                rank = tmp
        ranks[i] = rank
        top1[i] = inds_np[i][0]
        
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # R@1
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)  # R@5
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) # R@10
    
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr), (ranks, top1), inds_np
    
def eval(args):
    
    #load data
    image_embeddings = np.load('/shared/kgcoe-research/mil/new_cvs_data/img_features/test/flickr8k_test_r152_precomp.npy')
    image_embeddings = np.float32(image_embeddings)
    image_embeddings_rep = np.repeat(image_embeddings, repeats = 5, axis = 0)
    text_embeddings = np.load('/shared/kgcoe-research/mil/new_cvs_data/setnence_features/test/flickr8k_sentence_skipthoughts_test.npy')

    # pdb.set_trace()
    dataset = tf.data.Dataset.from_tensor_slices((image_embeddings_rep, text_embeddings))
    
    #Repeat for num epochs
    dataset = dataset.repeat(args.num_epochs)
    
    #Create the batch size
    dataset = dataset.batch(args.batch_size)
    
    #create iterator
    iterator = dataset.make_one_shot_iterator()
    
    #im_emb and txt_emb will be the image and txt having number samples = batch size 
    im_emb, txt_emb = iterator.get_next()
    
    image_embeddings_val=np.zeros((5000, args.emb_dim))
    text_embeddings_val=np.zeros((5000, args.emb_dim))
    model = CMR()
    image_embeddings_t, text_embeddings_t = model.build_rvs_model(im_emb,txt_emb, args, is_training = True)
        
    print(image_embeddings_t.shape)
    print(text_embeddings_t.shape)
    pdb.set_trace()
    # norm_image_embeddings = tf.nn.l2_normalize(tf.squeeze(image_embeddings_t), axis=1, name="norm_image_embeddings")
    # norm_text_embeddings = tf.nn.l2_normalize(text_embeddings_t, axis=1, name="norm_text_embeddings")
    saver=tf.train.Saver()       
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    
    with tf.Session(config=config) as sess:
        sess.run(tf.tables_initializer())
        saver.restore(sess, args.checkpoint)
        start_time = time.time()
        for i in range(0, 5000, args.batch_size):
            if i%100==0: print "Processed: {}".format(i)
            try:
                ie, te = sess.run([image_embeddings_t, text_embeddings_t])
                # pdb.set_trace()
                image_embeddings_val[i:i+args.batch_size, :] = np.squeeze(ie)
                text_embeddings_val[i:i+args.batch_size, :] = np.squeeze(te)

            except tf.errors.OutOfRangeError:
                break
    # pdb.set_trace()
    r1, r5, r10 = 0., 0., 0.
    # Average over 5 folds
    
    results=[]
    ri, ri0, i2t_ranked_idx = i2t_gpu(image_embeddings_val, text_embeddings_val, measure=args.measure)
    print "Image to Text: "
    print "R@1: {} R@5: {} R@10 : {} Med: {} Mean: {}".format(ri[0], ri[1], ri[2], ri[3], ri[4])
    # pdb.set_trace()
    rt, rt0, t2i_ranked_idx = t2i_gpu(image_embeddings_val, text_embeddings_val, measure=args.measure)
    print "Text to Image: "
    print "R@1: {} R@5: {} R@10 : {} Med: {} Mean: {}".format(rt[0], rt[1], rt[2], rt[3], rt[4])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--dataset', type=str, default='flickr8k', help="Type of dataset")
    # parser.add_argument('--num', type=int, default=None, help="Number of examples to be evaluated")
    # parser.add_argument('--stride', type=int, default=4, help="Value of stride in HRNE")
    # parser.add_argument('--max_len', type=int, default=None, help="Value of maximum caption length")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs to be evaluated")
    parser.add_argument('--emb_dim', type=int, default=512, help="Batch size")
    # parser.add_argument('--word_dim', type=int, default=300, help="Word Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0, help="dropout")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for Cross validation")
    parser.add_argument('--margin', type=float, default=0.05, help="Margin for sim loss")
    parser.add_argument('--precompute', action='store_true', help="Flag to use precomputed CNN features")
    # parser.add_argument('--num_units', type=int, default=1024, help="Number of hidden RNN units")
    # parser.add_argument('--vocab_size', type=int, default=26375, help="Number of hidden RNN units")
    # parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in RNN network")
    # parser.add_argument('--vocab_file', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc', help="Val file")
    # parser.add_argument('--val_ids_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/test.ids', help="Test IDs path")
    # parser.add_argument('--val_caps_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/test_caps.txt', help="Test captions path")
    # parser.add_argument('--test_sample', type=str, default='COCO_val2014_000000483108.jpg', help="Test captions path")
    parser.add_argument('--measure', type=str, default='cosine', help="Type of measure")
    # parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/coco_val_precompute.tfrecord', help="Path to val tfrecord")
    # parser.add_argument('--root_path', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014', help="Experiment dir")
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/new_cvs_data/experiment/model.ckpt', help="checkpoint")
    # parser.add_argument('--model', type=str, default='vse', help="Name of the model")
    # parser.add_argument('--mode', type=str, default='val', help="Training or validation")
    # parser.add_argument('--base', type=str, default='resnet_v2_152', help="Base architecture")
    # parser.add_argument('--use_abs', action='store_true', help="use_absolute values for embeddings")
    # parser.add_argument('--finetune_with_cnn', action='store_true', help="use_absolute values for embeddings")
    # parser.add_argument('--retrieve_text', action='store_true', help="Retrieve text given query image")
    # parser.add_argument('--retrieve_image', action='store_true', help="Retrieve image given query text")
    # parser.add_argument('--use_5_fold', action='store_true', help="Visualizing retrievals")
    args = parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
    	print key, ' : ', value
    print '--------------------------------'
    eval(args)
    
    
    
    