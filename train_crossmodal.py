import tensorflow as tf
import numpy as np
import argparse
from model import *
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import pdb
import datetime
import time
# tf.enable_eager_execution()
tf.set_random_seed(1234)
from sklearn.preprocessing import MinMaxScaler


def get_training_op(loss, args):
    """
    Defines the optimizers and returns the training op
    """
    # Gather all the variables in the graph
    
    all_vars = tf.trainable_variables()
    # pdb.set_trace()
    
    # Global step for the graph
    global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph())

    INITIAL_LEARNING_RATE=args.lr
    tf.summary.scalar('learning rate', INITIAL_LEARNING_RATE)
    # Define the optimizers. Here, feature extractor and metric embedding layers have different learning rates during training.
    if args.optimizer=='adam':
        optimizer_emb = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE)
    elif args.optimizer=='momentum':
        optimizer_emb = tf.train.MomentumOptimizer(learning_rate=INITIAL_LEARNING_RATE, momentum=0.98)
    
    # Get variables of specific sub networks using scope names
    vars_ie = get_vars(all_vars, scope_name='image_embedding', index=0)
    vars_te = get_vars(all_vars, scope_name='text_embedding', index=0)

    ie_len, te_len = len(vars_ie.values()), len(vars_te.values())
    
    # Calculate gradients for respective layers
    grad = tf.gradients(loss, vars_ie.values() + vars_te.values())
    grad_ie = grad[:ie_len]
    grad_te = grad[ie_len:ie_len+te_len]

    # Apply the gradients, update ops for batchnorm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer_emb.apply_gradients(zip(grad_ie+grad_te, vars_ie.values() + vars_te.values()), global_step=global_step)

    # return train_op, image_pretrain_saver, lstm_pretrain_saver,  global_step
    return train_op,global_step

def get_vars(all_vars, scope_name, index):
	"""
	Helper function used to return specific variables of a subgraph
	Args: 
		all_vars: All trainable variables in the graph
		scope_name: Scope name of the variables to retrieve
		index: Clip the variables in the graph at this index
	Returns:
		Dictionary of pre-trained checkpoint variables: new variables
	"""
	ckpt_vars = [var for var in all_vars if var.op.name.startswith(scope_name)]
	ckpt_var_dict = {}
	for var in ckpt_vars:
		actual_var_name  = var.op.name
		if actual_var_name.find('Logits') ==-1:
			clip_var_name = actual_var_name[index:]
			ckpt_var_dict[clip_var_name] = var
		
	return ckpt_var_dict



def train(args):
    
    
    #Reading in all the pre extracted features
    image_embeddings = np.load('/shared/kgcoe-research/mil/new_cvs_data/img_features/flickr8k_train_features.npy')
    image_embeddings = np.float32(image_embeddings)
    
    #since images are only 6k for training and each image has 5 captions, we have to repeat every image 5 times. 
    image_embeddings_rep = np.repeat(image_embeddings, repeats = 5, axis = 0)
    
    text_embeddings = np.load('/shared/kgcoe-research/mil/new_cvs_data/sentence_features/flickr8k_train_sentence_features.npy')
    
    audio_embeddings = np.load('/shared/kgcoe-research/mil/new_cvs_data/audio_features/flickr8k_train_audio_features.npy')
    audio_embeddings = np.float32(audio_embeddings)
    
    #Scaling the audio features
    scalar = MinMaxScaler(feature_range=(-1, 1))
    audio_embeddings = scalar.fit_transform(audio_embeddings)
    
    #Creating the iterator from the tf.data.Dataset
    #Feed npy file
    dataset = tf.data.Dataset.from_tensor_slices((image_embeddings_rep, text_embeddings))
    
    #Repeat for num epochs
    dataset = dataset.repeat(args.num_epochs)
    
    #Create the batch size
    dataset = dataset.batch(args.batch_size)
    
    #create iterator
    iterator = dataset.make_one_shot_iterator()
    
    #im_emb and txt_emb will be the image and txt having number samples = batch size 
    im_emb, txt_emb = iterator.get_next()
    
    #Build the CMR Model.
    model = CMR()
    ie, te = model.build_rvs_model(im_emb,txt_emb, args, is_training = True)
    # pdb.set_trace()
    # loss, loss_s, loss_im = model.sim_loss(image_placeholder, text_placeholder, args)
    loss, loss_s, loss_im = model.sim_loss(ie, te, args)
    total_loss = loss 

    #Get the training op
    
    train_op, global_step = get_training_op(total_loss, args)
    
    tf.summary.scalar('Sentence Loss', loss_s)
    tf.summary.scalar('Image Loss', loss_im)
    tf.summary.scalar('Total Loss', total_loss)
    summary_tensor = tf.summary.merge_all()
    summary_dir_name = '/shared/kgcoe-research/mil/new_cvs_data/experiment_new'
    checkpoint_dir_name = '/shared/kgcoe-research/mil/new_cvs_data/experiment_new'
    summary_filewriter = tf.summary.FileWriter(summary_dir_name, tf.get_default_graph())
    # pdb.set_trace()

    # checkpoint_saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.05, max_to_keep=0)
    # checkpoint_saver_hook = tf.train.CheckpointSaverHook(saver=checkpoint_saver, checkpoint_dir=checkpoint_dir_name, save_steps=args.save_steps)
    saver = tf.train.Saver(max_to_keep=2)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # pdb.set_trace()
    
    with tf.Session(config=session_config) as sess:
        # pdb.set_trace()
        sess.run([tf.global_variables_initializer()])
        # sess.run([iterator.initializer])
        param_file = open(os.path.join(checkpoint_dir_name, 'exp_params.txt'), 'w')
        for key, value in vars(args).items():
            param_file.write(str(key)+' : '+ str(value)+'\n')
        param_file.close()
        start_time = time.time()
        i=0
        
        while True:
            try:
                # features,labels = iterator.get_next()
                summary, _, loss, s_loss, im_loss, g_step, img, txt = sess.run([summary_tensor, train_op, total_loss, loss_s, loss_im, global_step, ie, te])
                if (g_step+1)%200 == 0:
                    print "Iteration : {} Total: {} Sentence : {} Image : {} ".format(g_step+1, loss, s_loss, im_loss)
                    summary_filewriter.add_summary(summary, g_step)
            except tf.errors.OutOfRangeError:
                break
            if ((g_step+1)%1000 == 0):
                print('Saving checkpoint at step %d' % (g_step+1))
                saver.save(sess, os.path.join(checkpoint_dir_name, 'model.ckpt'))
        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--measure', type=str, default='cosine', help="Type of loss")
    parser.add_argument('--saveEvery', type=int, default=100, help="How often checkpoints to save")
    
    parser.add_argument('--mine_n_hard', type=int, default=0, help="Flag to enable hard negative mining")
    parser.add_argument('--use_abs', action='store_true', help="use_absolute values for embeddings")
    parser.add_argument('--margin', type=float, default=0.05, help="Margin component")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--save_steps', type=int, default=200, help="Checkpoint saving step interval")
    
    #All the below arguments are for get_train_op
    parser.add_argument('--batch_size', type=int, default=120, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs")
    
    parser.add_argument('--decay_steps', type=int, default=10000, help="Checkpoint saving step interval")
    parser.add_argument('--decay_factor', type=float, default=0.9, help="Checkpoint saving step interval")
    # parser.add_argument('--emb_dim', type=int, default=512, help="CVS dimension")
    # parser.add_argument('--word_dim', type=int, default=300, help="Word Embedding dimension")
    
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    # parser.add_argument('--train_only_emb', action='store_true', help="train only embedding layer")
    # parser.add_argument('--no_train_cnn', action='store_true', help="Flag to not train CNN")
    # parser.add_argument('--no_pretrain_lstm', action='store_true', help="Flag to not use pre-trained LSTM weights")
    # parser.add_argument('--clip_grad_norm', type=float, default=None, help="Value of gradient clipping")
    # parser.add_argument('--precompute', action='store_true', help="Flag to use precomputed CNN features")
    

    
    args=parser.parse_args()
    train(args)
    
    
    
