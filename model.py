from keras.models import Sequential, Model

from keras.layers import LSTM, Embedding, Input, Reshape

import numpy as np

lstm_dim = 32

max_sentence_length = 5
max_word_root_length = 4
max_n_analyses = 10
max_analysis_length = 6
# max_sentence_length = 100

char_embedding_dim = 3
char_vocabulary_size = 29

tag_embedding_dim = 3
tag_vocabulary_size = 11

char_lstm_dim = 7

sentences_word_root_input = Input(shape=(max_sentence_length, max_word_root_length,),
                                  dtype='int32',
                                  name='sentences_word_root_input')

sentences_analysis_input = Input(shape=(max_sentence_length, max_n_analyses, max_analysis_length,),
                                  dtype='int32',
                                  name='sentences_analysis_input')

char_embedding_layer = Embedding(char_vocabulary_size,
                                 char_embedding_dim,
                                 input_length=max_word_root_length,
                                 name='char_embedding_layer')

tag_embedding_layer = Embedding(tag_vocabulary_size,
                                 tag_embedding_dim,
                                 input_length=max_analysis_length,
                                 name='tag_embedding_layer')

print sentences_word_root_input

input_char_embeddings = char_embedding_layer(sentences_word_root_input)

print input_char_embeddings

input_tag_embeddings = tag_embedding_layer(sentences_analysis_input)

char_lstm_layer = LSTM(char_lstm_dim,
                       input_shape=(None, max_sentence_length*max_word_root_length, char_embedding_dim))

from keras import backend as K

# rr = K.reshape(input_char_embeddings, [-1, max_sentence_length*max_word_root_length, char_embedding_dim])
r = Reshape((max_sentence_length*max_word_root_length, char_embedding_dim))
rr = r(input_char_embeddings)
char_lstm_layer_output = char_lstm_layer(rr)

print char_lstm_layer_output

model = Model(inputs=[sentences_word_root_input, sentences_analysis_input],
              outputs=[input_char_embeddings, char_lstm_layer_output])

# model = Model(inputs=[sentences_word_root_input, sentences_analysis_input],
#               outputs=[input_char_embeddings])

sample_input = [np.random.randint(0, char_vocabulary_size, max_sentence_length * max_word_root_length).reshape([-1, max_sentence_length, max_word_root_length]), \
                np.random.randint(0, tag_vocabulary_size, max_sentence_length * max_n_analyses * max_analysis_length).reshape([-1, max_sentence_length, max_n_analyses, max_analysis_length])]

p, _ = model.predict(sample_input)
print p
print p[0].shape

_, c = model.predict(sample_input)

# training
# model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# model = Sequential()
#
# model.add()
#
# model.add(LSTM(lstm_dim, input_shape=(-1, 64)))