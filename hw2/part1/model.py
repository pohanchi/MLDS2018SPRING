import numpy as np
import pickle
import tensorflow as tf


def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return lr_rate, keep_prob


def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.float32, [None, None, None], name='input')
    input_video_number = tf.placeholder(
        tf.int32, [None], name='input_video_number')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    target_sequence_length = tf.placeholder(
        tf.int32, [None], name='target_sequence_length')

    max_target_len = tf.reduce_max(target_sequence_length)
    return inputs, targets, target_sequence_length, max_target_len, input_video_number


def encoding_layer(batch_size, encoder_inputs, dim_video_feat, rnn_size,
                   num_layers, keep_prob, encoding_embedding_size,
                   max_encoder_steps):
    """
    :return: tuple (RNN output, RNN state)
    """

    encoder_inputs_flatten = tf.reshape(encoder_inputs, [-1, dim_video_feat])
    encoder_inputs_embedded = tf.layers.dense(
        encoder_inputs_flatten,
        encoding_embedding_size,
        use_bias=True,
        name='Encoding_Layer')
    embed = tf.reshape(encoder_inputs_embedded,
                       [batch_size, max_encoder_steps, rnn_size])

    stacked_cells = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)
        for _ in range(num_layers)
    ])

    outputs, state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)

    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer 
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        dec_cell, output_keep_prob=keep_prob)

    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                               target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True, maximum_iterations=max_summary_length)
    return outputs, _


def decoding_layer_infer(encoder_state,
                         dec_cell,
                         dec_embeddings,
                         start_of_sequence_id,
                         end_of_sequence_id,
                         max_target_sequence_length,
                         vocab_size,
                         output_layer,
                         batch_size,
                         keep_prob,
                         beam_size,
                         beam_search_mode=False):
    """
    Create a inference process in decoding layer 
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        dec_cell, output_keep_prob=keep_prob)

    if beam_search_mode == True:

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=dec_cell,
            embedding=dec_embeddings,
            start_tokens=tf.fill([int(batch_size / beam_size)],
                                 start_of_sequence_id),
            end_token=end_of_sequence_id,
            initial_state=encoder_state,
            beam_width=beam_size,
            output_layer=output_layer)
    else:

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            dec_embeddings, tf.fill([batch_size], start_of_sequence_id),
            end_of_sequence_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper,
                                                  encoder_state, output_layer)

    outputs, states, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=max_target_sequence_length)
    return outputs, states


def decoding_layer(dec_input, enc_outputs, encoder_state, input_video_number,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size, num_layers, target_vocab_to_int,
                   target_vocab_size, origin_batch_size, keep_prob,
                   decoding_embedding_size, beam_size, attention_mode,
                   beam_search_mode):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(
        tf.random_uniform([target_vocab_size, decoding_embedding_size]))

    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    if attention_mode == True:
        cells, encoder_state = attention(origin_batch_size, rnn_size,
                                         enc_outputs, encoder_state,
                                         input_video_number, cells)

    output_layer = tf.layers.Dense(
        target_vocab_size,
        kernel_initializer=tf.truncated_normal_initializer(
            mean=0.0, stddev=0.1, seed=9487))

    with tf.variable_scope("decode"):
        train_output, train_states = decoding_layer_train(
            encoder_state, cells, dec_embed_input, target_sequence_length,
            max_target_sequence_length, output_layer, keep_prob)
    batch_size = origin_batch_size
    if beam_search_mode == True:
        print("Using beamsearch decoding...")
        #enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=beam_size)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_size)
        # input_video_number = tf.contrib.seq2seq.tile_batch(input_video_number, multiplier=beam_size)
        batch_size = origin_batch_size * beam_size

    with tf.variable_scope("decode", reuse=True):
        infer_output, infer_states = decoding_layer_infer(
            encoder_state, cells, dec_embeddings, target_vocab_to_int['<bos>'],
            target_vocab_to_int['<eos>'], max_target_sequence_length,
            target_vocab_size, output_layer, batch_size, keep_prob, beam_size,
            beam_search_mode)

    return train_output, infer_output, infer_states


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<bos>' id
    go_id = target_vocab_to_int['<bos>']

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1],
                                   [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat


def seq2seq_model(input_data,
                  target_data,
                  keep_prob,
                  enc_batch_size,
                  input_video_number,
                  target_sequence_length,
                  dim_video_feat,
                  max_target_sentence_length,
                  target_vocab_size,
                  enc_embedding_size,
                  dec_embedding_size,
                  rnn_size,
                  num_layers,
                  target_vocab_to_int,
                  max_encoder_steps,
                  beam_size=0,
                  attention_mode=False,
                  beam_search_mode=False):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(
        enc_batch_size, input_data, dim_video_feat, rnn_size, num_layers,
        keep_prob, enc_embedding_size, max_encoder_steps)
    batch_size = enc_batch_size

    dec_input = process_decoder_input(target_data, target_vocab_to_int,
                                      enc_batch_size)

    train_output, infer_output, infer_states = decoding_layer(
        dec_input, enc_outputs, enc_states, input_video_number,
        target_sequence_length, max_target_sentence_length, rnn_size,
        num_layers, target_vocab_to_int, target_vocab_size, batch_size,
        keep_prob, dec_embedding_size, beam_size, attention_mode,
        beam_search_mode)

    return train_output, infer_output, infer_states


def attention(batch_size, rnn_size, encoder_outputs, encoder_state,
              encoder_inputs_length, decoder_cell):

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units=rnn_size,
        memory=encoder_outputs,
        normalize=True,
        memory_sequence_length=encoder_inputs_length)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=decoder_cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=rnn_size,
        name='Attention_Wrapper')
    decoder_initial_state = decoder_cell.zero_state(
        batch_size=batch_size,
        dtype=tf.float32).clone(cell_state=encoder_state)

    return decoder_cell, decoder_initial_state
