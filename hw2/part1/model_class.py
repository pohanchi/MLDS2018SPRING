import numpy as np
import pickle
import tensorflow as tf


class Seq2Seq_Model():
    def __init__(self, learning_rate, keep_prob, batch_size, dim_video_feat,
                 embedding_dim, max_encoder_steps, rnn_dim, num_layer,
                 target_data, target_vocab_to_int, dec_embedding_dim):
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.input_interface()

        self.batch_size = batch_size
        self.dim_video_feat = dim_video_feat
        self.max_encoder_steps = max_encoder_steps
        self.rnn_dim = rnn_dim
        self.num_layer = num_layer

        self.embedding_dim = embedding_dim

        self.input_interface()
        self.encoding_embedding()
        self.encoder_rnns()

        self.decoder_input_function(target_data, target_vocab_to_int)
        self.decoder_embedding_function(target_vocab_to_int, dec_embedding_dim)
        self.decoder_rnns()

        return

    def input_interface(self):
        self.videoes = tf.placeholder(
            tf.float32, [None, None, None], name='Video_Input')
        self.frame_number = tf.placeholder(
            tf.int32, [None], name='Frame_Number')

        self.Captions = tf.placeholder(tf.int32, [None, None], name='Caption')
        self.Caption_len = tf.placeholder(
            tf.int32, [None], name='Each_Caption_Length')

        self.Max_target_len = tf.reduce_nax(self.Caption_len)
        return

    def encoding_embedding(self):
        flatten = tf.reshape(self.videoes, [-1, self.dim_video_feat])
        self.flat_embed = tf.layers.dense(flatten, self.embedding_dim)
        return

    def encoder_rnns(self):
        embed = tf.reshape(
            self.flat_embed,
            [self.batch_size, self.max_encoder_steps, self.rnn_dim])
        encoder_cells = tf.contrib.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(self.rnn_dim, name='encoder_CELL'),
            self.keep_prob)
        two_encoder_layer_cells = tf.contrib.rnn.MultiRNNCell(
            [encoder_cells for _ in range(self.num_layer)])
        self.enc_outputs, self.enc_states = tf.nn.dynamic_rnn(
            two_encoder_layer_cells, embed, dtype=tf.float32)
        return

    def decoder_input_function(
            self,
            target_data,
            target_vocab_to_int,
    ):
        bos_id = target_vocab_to_int['<bos>']
        after_slice = tf.strided_slice(target_data, [0, 0],
                                       [self.batch_size, -1], [1, 1])
        self.dec_input = tf.concat(
            [tf.fill([self.batch_size, 1], bos_id), after_slice], 1)
        return

    def decoder_embedding_function(self, target_vocab_to_int,
                                   dec_embedding_dim):
        target_vocab_size = len(target_vocab_to_int)
        dec_embeddings_init = tf.Variable(
            tf.random_uniform([target_vocab_size, dec_embedding_dim]))
        self.dec_embeddiing = tf.nn.embedding_lookup(dec_embeddings_init,
                                                     self.dec_input)

    def decoder_rnns(self):
        decoder_cell = tf.contrib.rnn.LSTMCell(
            self.rnn_dim, name='decoder_CELL')
        self.two_layer_dec_cells = tf.contrib.rnn.MultiRNNCell(
            [decoder_cell for _ in range(self.num_layer)])
        return

    def train_mode(self):
        return

    def infer_mode(self):
        return

    def attention(self):
        return

    def beam_search(self):
        return

    def schedule_sample(self):
        return
