
import argparse
import os

from loader import encode_sentence, read_datafile
from model import build_model, Params

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--command", default="train",required=True, choices=["train", "predict"])
parser.add_argument("--train_filepath", required=True)
parser.add_argument("--test_filepath", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--model_path")

args = parser.parse_args()


def sample_generator(sentences, label2ids, batch_size=32):
    while True:
        n_in_batch = 0
        batch = [[], []]
        shuffled_indices = np.random.permutation(len(sentences))
        for sentence_i in shuffled_indices:
            sentence = sentences[sentence_i]
            sentences_word_root_input, sentences_analysis_input, surface_form_input, correct_tags_input = \
                encode_sentence(sentence, label2ids)

            input_array = []
            for array in [sentences_word_root_input, sentences_analysis_input, surface_form_input,
                          correct_tags_input]:
                # print array.shape
                # print array.reshape([-1] + list(array.shape)).shape
                input_array += [np.expand_dims(np.copy(array), axis=0)]
                # print array.shape
                # print np.expand_dims(array, axis=0).shape

            batch[0].append(input_array[:-1])
            batch[1].append([input_array[-1]])
            n_in_batch += 1
            if n_in_batch == batch_size:
                # yield np.concatenate(batch[0], axis=0), np.concatenate(batch[1], axis=0)
                # for b in batch[1]:
                #     for i in range(1):
                #         print i
                #         print b[i].shape
                yield [np.concatenate([b[i] for b in batch[0]], axis=0) for i in range(3)], \
                      [np.concatenate([b[0] for b in batch[1]], axis=0)]
                # yield batch[0], batch[1]
                n_in_batch = 0
                batch = [[], []]
        if n_in_batch > 0:
            yield [np.concatenate([b[i] for b in batch[0]], axis=0) for i in
                   range(3)], \
                  [np.concatenate([b[0] for b in batch[1]], axis=0)]

            # yield (input_array[:-1], input_array[-1])


train_and_test_sentences, label2ids = read_datafile(args.train_filepath, args.test_filepath)

params = Params()

params.max_sentence_length = label2ids['max_sentence_length']
params.max_n_analyses = label2ids['max_n_analysis']
params.batch_size = 1
params.max_surface_form_length = label2ids['max_surface_form_length']
params.max_word_root_length = label2ids['max_word_root_length']
params.max_analysis_length = label2ids['max_analysis_length']

params.char_vocabulary_size = label2ids['character_unique_count']['value']
params.tag_vocabulary_size = label2ids['morph_token_unique_count']['value']

params.char_lstm_dim = 5
params.char_embedding_dim = 5

params.tag_lstm_dim = params.char_lstm_dim
params.tag_embedding_dim = 5

params.sentence_level_lstm_dim = 2 * params.char_lstm_dim

if args.command == "train":

    # train_and_test_sentences, label2ids = read_datafile("test.merge.utf8", "test.merge.utf8")

    model = build_model(params)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

    checkpointer = ModelCheckpoint(filepath="./models/ntd-{run_name}".format(run_name=args.run_name)
                                            + '.epoch-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5',
                                   verbose=1,
                                   monitor="val_acc",
                                   save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=os.path.join('./logs', args.run_name),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                embeddings_freq=1,
                embeddings_layer_names=None,
                embeddings_metadata={'char_embedding_layer': 'char_embedding_layer.tsv',
                                     'tag_embedding_layer': 'tag_embedding_layer.tsv'})
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
                                  verbose=0,
                                  mode='auto',
                                  epsilon=0.0001,
                                  cooldown=1,
                                  min_lr=0)
    model.fit_generator(sample_generator(train_and_test_sentences[0], label2ids, batch_size=params.batch_size),
                        steps_per_epoch=len(train_and_test_sentences[0])/params.batch_size,
                        epochs=10,
                        validation_data=sample_generator(train_and_test_sentences[1], label2ids, batch_size=params.batch_size),
                        validation_steps=len(train_and_test_sentences[1])/params.batch_size+1,
                        callbacks=[checkpointer, tensorboard_callback, reduce_lr])

    print "Saving"
    model.save("./models/ntd-{run_name}-final.hdf5".format(run_name=args.run_name))
    # model.evaluate()

elif args.command == "predict":

    from keras.models import load_model

    from model import build_model

    model = build_model(params)

    assert args.model_path, "--model_path should be given in the arguments for 'predict'"

    model.load_weights(args.model_path)

    for sample in sample_generator(train_and_test_sentences[1], label2ids, batch_size=params.batch_size):
        print sample
        pred_probs = model.predict(sample[0], batch_size=params.batch_size, verbose=1)
        print pred_probs
        print pred_probs.shape
        print np.argmax(pred_probs, axis=2)
        print sample[1][0]
        print sample[1][0].shape
        print np.argmax(sample[1][0], axis=2)
        sys.exit()