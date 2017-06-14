from loader import encode_sentence, read_datafile
from model import build_model, Params

import numpy as np

sentences, label2ids = read_datafile("sample.data")

params = Params()

params.max_sentence_length = label2ids['max_sentence_length']
params.max_n_analyses = label2ids['max_n_analysis']
params.batch_size = 1
params.max_surface_form_length = label2ids['max_surface_form_length']
params.max_word_root_length = label2ids['max_word_root_length']
params.max_analysis_length = label2ids['max_analysis_length']

params.char_vocabulary_size = label2ids['character_unique_count']['value']
params.tag_vocabulary_size = label2ids['morph_token_unique_count']['value']

params.char_lstm_dim = 100
params.char_embedding_dim = 100

params.tag_lstm_dim = params.char_lstm_dim
params.tag_embedding_dim = 100

params.sentence_level_lstm_dim = 2 * params.char_lstm_dim

model = build_model(params)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

x_input = []
y_input = []


def sample_generator():

    while True:
        for sentence in sentences:
            sentences_word_root_input, sentences_analysis_input, surface_form_input, correct_tags_input = \
                encode_sentence(sentence, label2ids)

            input_array = []
            for array in [sentences_word_root_input, sentences_analysis_input, surface_form_input, correct_tags_input]:
                # print array.shape
                # print array.reshape([-1] + list(array.shape)).shape
                input_array += [array.reshape([-1] + list(array.shape))]

            #input_array = [sentences_word_root_input, sentences_analysis_input, surface_form_input]

            # x_input += [input_array[:-1]]
            # y_input += [input_array[-1]]
            yield (input_array[:-1], input_array[-1])

# model.fit(x_input,
#           y_input, batch_size=params.batch_size, epochs=10)

model.fit_generator(sample_generator(), steps_per_epoch=len(sentences), epochs=10,
                    validation_data=sample_generator(), validation_steps=len(sentences))



p_pred = model.predict(next(sample_generator())[0])
print p_pred



