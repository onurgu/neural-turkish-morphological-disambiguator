from keras.models import Model

from keras.layers import LSTM, Embedding, Input, Reshape,\
    TimeDistributed, Bidirectional, Add, Activation, Lambda, Dot, Layer

from keras.utils.np_utils import to_categorical

from keras import backend as K
import tensorflow as tf

import numpy as np

from maskedreshape import MaskedReshape


class Params():

    def __init__(self):
        pass

    max_sentence_length = 5
    max_word_root_length = 4
    max_n_analyses = 10
    max_analysis_length = 6

    max_surface_form_length = 13

    char_embedding_dim = 3
    char_vocabulary_size = 29

    tag_embedding_dim = 3
    tag_vocabulary_size = 11

    char_lstm_dim = 7
    tag_lstm_dim = char_lstm_dim
    assert tag_lstm_dim == char_lstm_dim

    sentence_level_lstm_dim = 2 * char_lstm_dim

    batch_size = 11

params = Params()

def dot_product_over_specific_axis(inputs):
    print "INPUTS TO LAMBDA:", inputs
    x = inputs[0]
    # x [?, 5, 10, 14]
    y = inputs[1]
    # x = tf.transpose(x, [0,1,3,2])
    # x.T [?, 5, 14, 10]
    # y [?, 5, 14]
    y = tf.reshape(y, tf.concat([tf.shape(y), [1]], axis=0))
    # y [?, 5, 14, 1]
    result = tf.matmul(x, y)
    result = tf.squeeze(result, axis=[3])
    return result

def fabricate_calc_output_shape(max_sentence_length, max_n_analyses):

    def calc_output_shape(input_shape):
        return tuple([None, max_sentence_length, max_n_analyses])

    return calc_output_shape

def create_context_bi_lstm(input_3d, embedding_layer,
                             max_sentence_length, max_surface_form_length,
                             lstm_dim, embedding_dim,
                             sentence_level_lstm_dim):
    r = Reshape((max_sentence_length * max_surface_form_length,))
    rr = r(input_3d)
    input_embeddings = embedding_layer(rr)
    print input_embeddings
    # input_embeddings = Lambda(lambda x: x, output_shape=lambda s: s)(input_embeddings)
    r = MaskedReshape((max_sentence_length, max_surface_form_length, embedding_dim),
                      (max_sentence_length, max_surface_form_length))
    rr = r(input_embeddings)
    lstm_layer = Bidirectional(LSTM(lstm_dim,
                                         input_shape=(max_surface_form_length, embedding_dim)))
    td_lstm_layer = TimeDistributed(lstm_layer,
                                         input_shape=(max_surface_form_length, embedding_dim))

    char_bi_lstm_outputs = td_lstm_layer(rr)
    print "char_bi_lstm_outputs", char_bi_lstm_outputs

    sentence_level_lstm_layer = Bidirectional(LSTM(sentence_level_lstm_dim,
                                                   input_shape=(max_sentence_length, 2 * lstm_dim),
                                                   return_sequences=True),
                                              merge_mode='sum',
                                              input_shape=(max_sentence_length, sentence_level_lstm_dim))
    # sentence_level_td_lstm_layer = TimeDistributed(sentence_level_lstm_layer,
    #                                      input_shape=(max_sentence_length, 2 * lstm_dim))
    # sentence_level_bi_lstm_outputs = sentence_level_td_lstm_layer(char_bi_lstm_outputs)
    char_bi_lstm_outputs = Lambda(lambda x: x, output_shape=lambda s: s)(char_bi_lstm_outputs)
    sentence_level_bi_lstm_outputs = sentence_level_lstm_layer(char_bi_lstm_outputs)
    sentence_level_bi_lstm_outputs_tanh = Activation('tanh')(sentence_level_bi_lstm_outputs)

    print "sentence_level_bi_lstm_outputs", sentence_level_bi_lstm_outputs
    print "sentence_level_bi_lstm_outputs_tanh", sentence_level_bi_lstm_outputs_tanh

    return sentence_level_bi_lstm_outputs_tanh



def create_two_level_bi_lstm(input_4d, embedding_layer,
                             max_sentence_length, max_n_analyses, max_word_root_length,
                             lstm_dim, embedding_dim):
    r = Reshape((max_sentence_length * max_n_analyses * max_word_root_length,))
    # input_4d = Lambda(lambda x: x, output_shape=lambda s: s)(input_4d)
    rr = r(input_4d)
    input_embeddings = embedding_layer(rr)
    print input_embeddings
    r = MaskedReshape((max_sentence_length * max_n_analyses, max_word_root_length, embedding_dim),
                      (max_sentence_length * max_n_analyses, max_word_root_length))
    # input_embeddings = Lambda(lambda x: x, output_shape=lambda s: s)(input_embeddings)
    rr = r(input_embeddings)
    lstm_layer = Bidirectional(LSTM(lstm_dim,
                                         input_shape=(max_word_root_length, embedding_dim)))
    td_lstm_layer = TimeDistributed(lstm_layer,
                                         input_shape=(max_word_root_length, embedding_dim))

    lstm_layer_output = td_lstm_layer(rr)
    lstm_layer_output_relu = Activation('relu')(lstm_layer_output)
    print "lstm_layer_output_relu", lstm_layer_output_relu
    r = Reshape((max_sentence_length, max_n_analyses, 2 * lstm_dim))
    lstm_layer_output_relu = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_layer_output_relu)
    lstm_layer_output_relu_reshaped = r(lstm_layer_output_relu)
    print "lstm_layer_output_relu_reshaped", lstm_layer_output_relu_reshaped
    return input_embeddings, lstm_layer_output_relu_reshaped


def build_model(params=Params()):

    sentences_word_root_input = Input(shape=(params.max_sentence_length, params.max_n_analyses, params.max_word_root_length),
                                      dtype='int32',
                                      name='sentences_word_root_input')

    sentences_analysis_input = Input(shape=(params.max_sentence_length, params.max_n_analyses, params.max_analysis_length,),
                                      dtype='int32',
                                      name='sentences_analysis_input')

    char_embedding_layer = Embedding(params.char_vocabulary_size+1,
                                     params.char_embedding_dim,
                                     #input_length=max_word_root_length,
                                     name='char_embedding_layer',
                                     mask_zero=True)

    tag_embedding_layer = Embedding(params.tag_vocabulary_size+1,
                                    params.tag_embedding_dim,
                                     #input_length=max_analysis_length,
                                     name='tag_embedding_layer',
                                    mask_zero=True)

    surface_form_input = Input(shape=(params.max_sentence_length, params.max_surface_form_length,),
                                      dtype='int32',
                                      name='surface_form_input')

    # correct_tag_input = Input(shape=(max_sentence_length,),
    #                                   dtype='int32',
    #                                   name='correct_tag_input')

    print sentences_word_root_input



    input_char_embeddings, char_lstm_layer_output = \
        create_two_level_bi_lstm(sentences_word_root_input, char_embedding_layer,
                                 params.max_sentence_length, params.max_n_analyses, params.max_word_root_length,
                                 params.char_lstm_dim, params.char_embedding_dim)

    input_tag_embeddings, tag_lstm_layer_output = \
        create_two_level_bi_lstm(sentences_analysis_input,
                                 tag_embedding_layer,
                                 params.max_sentence_length, params.max_n_analyses, params.max_analysis_length,
                                 params.tag_lstm_dim, params.tag_embedding_dim)

    print "char_lstm_layer_output", char_lstm_layer_output

    added_root_and_analysis_embeddings = Add()([char_lstm_layer_output, tag_lstm_layer_output])
    R_matrix = Activation('tanh')(added_root_and_analysis_embeddings)
    # (None, max_sentence_length, max_n_analyses, 2*char_lstm_dim)

    print "R_matrix", R_matrix

    h = create_context_bi_lstm(surface_form_input, char_embedding_layer,
                               params.max_sentence_length, params.max_surface_form_length,
                               params.char_lstm_dim, params.char_embedding_dim, params.sentence_level_lstm_dim)

    print "h", h

    ll = Lambda(dot_product_over_specific_axis,
                output_shape=fabricate_calc_output_shape(params.max_sentence_length, params.max_n_analyses))

    # compute h
    p = Activation('softmax', name="p")(ll([R_matrix, h]))

    print "p", p

    predicted_tags = K.max(p, axis=2)

    model = Model(inputs=[sentences_word_root_input, sentences_analysis_input, surface_form_input],
                  outputs=[p])
    return model

if __name__ == "__main__":

    params = Params()
    model = build_model(params)

    # training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    sample_correct_tags = np.expand_dims(to_categorical(np.random.randint(0, params.max_n_analyses, params.max_sentence_length).reshape([-1, params.max_sentence_length]), params.max_n_analyses),
                                         axis=0)

    print sample_correct_tags

    sample_input = [np.random.randint(0, params.char_vocabulary_size+1, params.max_sentence_length *  params.max_n_analyses * params.max_word_root_length).reshape([-1, params.max_sentence_length, params.max_n_analyses, params.max_word_root_length]), \
                    np.random.randint(0, params.tag_vocabulary_size+1, params.max_sentence_length * params.max_n_analyses * params.max_analysis_length).reshape([-1, params.max_sentence_length, params.max_n_analyses, params.max_analysis_length]), \
                    np.random.randint(0, params.char_vocabulary_size+1, params.max_sentence_length * params.max_surface_form_length).reshape([-1, params.max_sentence_length, params.max_surface_form_length]),
                    ]

    model.fit(sample_input, sample_correct_tags, batch_size=1, epochs=10)

    p_pred = model.predict(sample_input)

    print p_pred

    # p, _ = model.predict(sample_input)
    # print p
    # print p[0].shape

    #
    # model.fit()

    # model = Sequential()
    #
    # model.add()
    #
    # model.add(LSTM(lstm_dim, input_shape=(-1, 64)))