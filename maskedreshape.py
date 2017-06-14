from keras import backend as K
import numpy as np

from keras.models import Model

from keras.layers import Reshape, Input, Bidirectional, TimeDistributed, Embedding, LSTM, Activation, Lambda

from keras.utils.np_utils import to_categorical

class MaskedReshape(Reshape):
    def __init__(self, target_shape, target_mask_shape, **kwargs):
        self.supports_masking = True
        self.target_mask_shape = target_mask_shape
        super(MaskedReshape, self).__init__(target_shape, **kwargs)

    def call(self, inputs, mask=None):
        # In case the target shape is not fully defined,
        # we need access to the shape of x.
        # solution:
        # 1) rely on x._keras_shape
        # 2) fallback: K.int_shape
        target_shape = self.target_shape
        target_mask_shape = self.target_mask_shape
        if -1 in target_shape:
            # target shape not fully defined
            input_shape = None
            try:
                input_shape = K.int_shape(inputs)
            except TypeError:
                pass
            if input_shape is not None:
                target_shape = self.compute_output_shape(input_shape)[1:]

        _result = K.reshape(inputs, (-1,) + target_shape)
        reshaped_mask = K.reshape(mask, (-1,) + target_mask_shape + (1,))
        result = _result * K.cast(reshaped_mask, K.floatx())
        return result

    def compute_mask(self, inputs, mask=None):
        target_shape = self.target_shape
        target_mask_shape = self.target_mask_shape
        if -1 in target_shape:
            # target shape not fully defined
            input_shape = None
            try:
                input_shape = K.int_shape(inputs)
            except TypeError:
                pass
            if input_shape is not None:
                target_shape = self.compute_output_shape(input_shape)[1:]
        return K.reshape(mask, (-1,) + target_mask_shape)

if __name__ == "__main__":
    print "deneme"

    max_sentence_length = 5
    max_word_root_length = 4
    max_n_analyses = 10
    max_analysis_length = 6

    char_embedding_dim = 3
    char_vocabulary_size = 29

    char_lstm_dim = 7

    sentences_word_root_input = Input(
        shape=(max_sentence_length, max_n_analyses, max_word_root_length),
        dtype='int32',
        name='sentences_word_root_input')

    char_embedding_layer = Embedding(char_vocabulary_size + 1,
                                     char_embedding_dim,
                                     # input_length=max_word_root_length,
                                     name='char_embedding_layer',
                                     mask_zero=True)


    def create_two_level_bi_lstm(input_4d, embedding_layer,
                                 max_sentence_length, max_n_analyses, max_word_root_length,
                                 lstm_dim, embedding_dim):
        r = Reshape((max_sentence_length * max_n_analyses * max_word_root_length,))
        # input_4d = Lambda(lambda x: x, output_shape=lambda s: s)(input_4d)
        rr = r(input_4d)
        input_embeddings = embedding_layer(rr)
        print input_embeddings
        r = MaskedReshape(
            (max_sentence_length * max_n_analyses, max_word_root_length, embedding_dim),
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

    input_char_embeddings, char_lstm_layer_output = \
        create_two_level_bi_lstm(sentences_word_root_input, char_embedding_layer,
                                 max_sentence_length, max_n_analyses, max_word_root_length,
                                 char_lstm_dim, char_embedding_dim)

    model = Model(inputs=[sentences_word_root_input],
                  outputs=[input_char_embeddings, char_lstm_layer_output])

    # training
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    sample_correct_tags = np.expand_dims(to_categorical(
        np.random.randint(0, max_n_analyses, max_sentence_length).reshape(
            [-1, max_sentence_length]), max_n_analyses),
                                         axis=0)

    print sample_correct_tags

    sample_input = [np.random.randint(0, char_vocabulary_size + 1,
                                      max_sentence_length * max_n_analyses * max_word_root_length)
        .reshape([-1, max_sentence_length, max_n_analyses, max_word_root_length])]

    print sample_input
    print sample_input[0].shape

    outputs = model.predict(sample_input, batch_size=1)

    print outputs

    # model.fit(sample_input, [sample_correct_tags, sample_correct_tags], batch_size=1, epochs=10)