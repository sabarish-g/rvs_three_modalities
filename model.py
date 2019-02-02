import tensorflow as tf
import numpy as np
from dnn_library import *
from nets import *
import numpy as np
import pdb
# from text_encoder import *
# from attention import *

class CMR(object):
    """
    Base class for Cross-Modal Retrieval experiments
    """  # '/shared/kgcoe-research/mil/peri/scan_data/mscoco_vocab.txt'
    # def __init__(self, base='inception_v1', vocab_file='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc', margin=1., embedding_dim=512,word_dim=1024, vocab_size=26735):
    def __init__(self, embedding_dim=512,margin=1.):
        self.scope_name='CMR'
        # self.base_arch = base
        self.margin=margin
        # self.vocab_file = vocab_file
        # self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # self.word_dim = word_dim
    
    
    def _build_embedding(self, feat_anchor, embedding_dim=512, scope_name="embedding", act_fn=tf.nn.tanh, reuse=None):
        """
        Build the embedding network
        """
        with slim.arg_scope([slim.fully_connected],
                             activation_fn=act_fn,
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(0.0002),
                             reuse=reuse): #
            embedding = slim.fully_connected(feat_anchor, embedding_dim, activation_fn=act_fn, scope=scope_name)

        return embedding	        
    
    def build_rvs_model(self, image_features, text_features, params, is_training=True, reuse=None):
        """
        Builds the VSE model
        """
        #Build the embeddings for images.
        image_embeddings = self._build_embedding(image_features, self.embedding_dim, act_fn=None, reuse=reuse, scope_name='image_embedding')
        
        #Build the embeddings for captions
        text_embeddings = self._build_embedding(text_features, self.embedding_dim, act_fn=None, reuse=reuse, scope_name='text_embedding')  

        return image_embeddings, text_embeddings
        

    def sim_loss(self, image_embeddings, text_embeddings, params, sim_scores=None):
        """
        Order violation or cosine similarity loss for image and text embeddings
        """
        
        with tf.name_scope('Sim_Loss') as scope:
            if sim_scores is None:
                norm_image_embeddings = tf.nn.l2_normalize(image_embeddings, axis=1, name="norm_image_embeddings")
                norm_text_embeddings = tf.nn.l2_normalize(text_embeddings, axis=1, name="norm_text_embeddings")
                
                if params.use_abs:
                    norm_image_embeddings = tf.abs(norm_image_embeddings)
                    norm_text_embeddings = tf.abs(norm_text_embeddings)
                    
                if params.measure=='cosine':                    
                    sim_scores = tf.matmul(norm_image_embeddings, norm_text_embeddings, transpose_b=True, name='sim_score')
                elif params.measure=='order':
                    # refer to eqn in paper or code of http://openaccess.thecvf.com/content_cvpr_2018/papers/Wehrmann_Bidirectional_Retrieval_Made_CVPR_2018_paper.pdf
                    im_emb = tf.expand_dims(norm_image_embeddings, 0) # 1x128x2048
                    text_emb = tf.expand_dims(norm_text_embeddings, 1) # 128x1x2048
                    im_emb = tf.tile(im_emb, [norm_image_embeddings.shape.as_list()[0], 1, 1]) # 128x128x2048 (Each row has 128x2048 im_emb)
                    text_emb = tf.tile(text_emb, [1, norm_text_embeddings.shape.as_list()[0], 1]) # 128x128x2048 (Each row has its text emb replicated 128 times)
                    sqr_diff = tf.square(tf.maximum(text_emb - im_emb, 0.)) 
                    sqr_diff_sum = tf.squeeze(tf.reduce_sum(sqr_diff, 2))
                    sim_scores = -tf.transpose(tf.sqrt(sqr_diff_sum), name='order_sim_scores')                                         
              
            # Get the diagonal of the matrix
            sim_diag = tf.expand_dims(tf.diag_part(sim_scores), 0, name='sim_diag')
            # sim_diag_tile = tf.tile(sim_diag, multiples=[sim_diag.shape.as_list()[1], 1], name='sim_diag_tile')
            sim_diag_tile = tf.tile(sim_diag, multiples=[120, 1], name='sim_diag_tile')
            sim_diag_transpose = tf.transpose(sim_diag, name='sim_diag_transpose')
            # sim_diag_tile_transpose = tf.tile(sim_diag_transpose, multiples=[1, sim_diag.shape.as_list()[1]], name='sim_diag_tile_transpose')
            sim_diag_tile_transpose = tf.tile(sim_diag_transpose, multiples=[1, 120], name='sim_diag_tile_transpose')

            # compare every diagonal score to scores in its column
            # caption retrieval
            loss_s = tf.maximum(self.margin + sim_scores - sim_diag_tile_transpose, 0.)
            # compare every diagonal score to scores in its row
            # image retrieval
            loss_im = tf.maximum(self.margin + sim_scores - sim_diag_tile, 0.)

            # clear the costs for diagonal elements
            # mask = tf.eye(loss_s.shape.as_list()[0], dtype=tf.bool, name='Mask')
            mask = tf.eye(120, dtype=tf.bool, name='Mask')
            mask_not = tf.cast(tf.logical_not(mask, name='mask_not'), tf.float32)
            
            neg_s_loss   = tf.multiply(loss_s, mask_not, name='neg_s_loss')
            neg_im_loss = tf.multiply(loss_im, mask_not, name='neg_im_loss')

            # Mining the hardest negative for each sample
            if params.mine_n_hard>0:
                if params.mine_n_hard==1:
                    loss_s = tf.reduce_max(neg_s_loss, axis=1)
                    loss_im = tf.reduce_max(neg_im_loss, axis=0)
                else:
                    loss_s = tf.contrib.framework.sort(neg_s_loss, axis=1, direction='DESCENDING')
                    loss_im = tf.contrib.framework.sort(neg_im_loss, axis=0, direction='DESCENDING')
                    # Build the index matrix to gather_nd
                    # batch_size=loss_s.shape.as_list()[0]
                    batch_size=120
                    indices= np.zeros((batch_size, mine_n_hard, 2))
                    for it in range(batch_size):
                        for m in range(mine_n_hard):
                            indices[it][m][0] = it
                            indices[it][m][1] = m

                    # Get the top N distances and reduce them
                    loss_s = tf.gather_nd(loss_s, indices.astype(np.int32))
                    loss_im = tf.gather_nd(loss_im, indices.astype(np.int32))
                
            loss_s = tf.reduce_sum(loss_s, name='loss_s')
            loss_im = tf.reduce_sum(loss_im, name='loss_im')
            
            total_loss = loss_s + loss_im
            
            return total_loss, loss_s, loss_im