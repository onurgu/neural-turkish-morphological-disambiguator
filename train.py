
import argparse
from collections import defaultdict as dd
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


def sample_generator(sentences, label2ids, batch_size=32, return_sentence=False):
    while True:
        n_in_batch = 0
        batch = [[], []]
        decoded_sentences_in_batch = []
        shuffled_indices = np.random.permutation(len(sentences))
        for sentence_i in shuffled_indices:
            sentence = sentences[sentence_i]
            sentences_word_root_input, sentences_analysis_input, surface_form_input, correct_tags_input = \
                encode_sentence(sentence, label2ids)

            if return_sentence:
                decoded_sentences_in_batch.append(sentence)

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
                encoded_sentences_in_batch = ([np.concatenate([b[i] for b in batch[0]], axis=0) for i in range(3)], \
                          [np.concatenate([b[0] for b in batch[1]], axis=0)])
                if not return_sentence:
                    yield encoded_sentences_in_batch
                else:
                    yield encoded_sentences_in_batch, decoded_sentences_in_batch
                # yield batch[0], batch[1]
                n_in_batch = 0
                batch = [[], []]
                decoded_sentences_in_batch = []

        if n_in_batch > 0:
            encoded_sentences_in_batch = ([np.concatenate([b[i] for b in batch[0]], axis=0) for i in
                   range(3)], \
                  [np.concatenate([b[0] for b in batch[1]], axis=0)])
            if not return_sentence:
                yield encoded_sentences_in_batch
            else:
                yield encoded_sentences_in_batch, decoded_sentences_in_batch


train_and_test_sentences, label2ids = read_datafile(args.train_filepath, args.test_filepath)

params = Params()

params.max_sentence_length = label2ids['max_sentence_length']
params.max_n_analyses = label2ids['max_n_analysis']

params.batch_size = 1
params.n_subepochs = 40

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

if args.command == "train":

    # train_and_test_sentences, label2ids = read_datafile("test.merge.utf8", "test.merge.utf8")

    model = build_model(params)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

    checkpointer = ModelCheckpoint(filepath="./models/ntd-{run_name}".format(run_name=args.run_name)
                                            + '.epoch-{epoch:02d}-val_acc-{val_acc:.5f}.hdf5',
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
                        steps_per_epoch=len(train_and_test_sentences[0])/params.batch_size/params.n_subepochs,
                        epochs=10*params.n_subepochs,
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

    total_correct = 0
    total_tokens = 0

    total_correct_all_structure = 0
    total_tokens_all_structure = 0

    total_correct_ambigious = 0
    total_tokens_ambigious = 0

    correct_counts = dd(int)
    total_counts = dd(int)


    for batch_idx, (sample_batch, decoded_sample_batch) in enumerate(sample_generator(train_and_test_sentences[1],
                                                               label2ids,
                                                               batch_size=params.batch_size,
                                                               return_sentence=True)):
        # print sample
        # pred_probs = model.predict(sample_batch[0], batch_size=params.batch_size, verbose=1)
        # print pred_probs
        # print pred_probs.shape
        # print np.argmax(pred_probs, axis=2)
        # print sample_batch[1][0]
        # print sample_batch[1][0].shape
        # print np.argmax(sample_batch[1][0], axis=2)
        # print decoded_sample_batch

        # print "decoded_sample_batch: ", decoded_sample_batch

        correct_tags = np.argmax(sample_batch[1][0], axis=2)
        pred_probs = model.predict(sample_batch[0], batch_size=params.batch_size, verbose=1)
        pred_tags = np.argmax(pred_probs, axis=2)

        for idx, correct_tag in enumerate(correct_tags):

            sentence_length = len(decoded_sample_batch[idx]['surface_form_lengths'])
            pred_tag = pred_tags[idx]
            # print correct_tag
            # print correct_tag.shape
            # print pred_tag
            # print pred_tag.shape
            n_correct = np.sum(correct_tag[:sentence_length] == pred_tag[:sentence_length])
            total_correct += n_correct
            total_tokens += sentence_length
            # print "sentence_length: ", sentence_length

            # print "n_correct: ", n_correct
            # print "sentence_length: ", sentence_length

            total_correct_all_structure += n_correct + correct_tag.shape[0] - sentence_length
            total_tokens_all_structure += correct_tag.shape[0]

            baseline_log_prob = 0
            import math

            for j in range(len(decoded_sample_batch[idx]['roots'])):
                if j >= sentence_length:
                    break
                n_analyses = len(decoded_sample_batch[idx]['roots'][j])
                baseline_log_prob += math.log(1/float(n_analyses))
                # print n_analyses
                assert n_analyses >= 1
                if n_analyses > 1:
                    if correct_tag[j] == pred_tag[j]:
                        total_correct_ambigious += 1
                        correct_counts[n_analyses] += 1
                    total_tokens_ambigious += 1
                    total_counts[n_analyses] += 1


        if batch_idx % 100 == 0:
            print "only the filled part of the sentence"
            print total_correct
            print total_tokens
            print float(total_correct)/total_tokens
            print "all the sentence"
            print total_correct_all_structure
            print total_tokens_all_structure
            print float(total_correct_all_structure)/total_tokens_all_structure
            print "==="
            print "ambigous"
            print total_correct_ambigious
            print total_tokens_ambigious
            print float(total_correct_ambigious)/total_tokens_ambigious
            print "==="
            for key in correct_counts:
                print "disambiguations out of n_analyses: %d ===> %lf" % (key, float(correct_counts[key])/total_counts[key])
            print "==="
        if batch_idx*params.batch_size >= len(train_and_test_sentences[1]):
            print "Evaluation finished, batch_id: %d" % batch_idx
            print "only the filled part of the sentence"
            print total_correct
            print total_tokens
            print float(total_correct)/total_tokens
            print "all the sentence"
            print total_correct_all_structure
            print total_tokens_all_structure
            print float(total_correct_all_structure) / total_tokens_all_structure
            print "==="
            print "ambigous"
            print total_correct_ambigious
            print total_tokens_ambigious
            print float(total_correct_ambigious)/total_tokens_ambigious
            print "==="
            for key in correct_counts:
                print "disambiguations out of n_analyses: %d ===> %lf %d %d" % (key, float(correct_counts[key])/total_counts[key], correct_counts[key], total_counts[key])
            print "==="
            break

