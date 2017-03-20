
import codecs

from collections import defaultdict as dd


def _encode_label(dictionary_name, label, label2ids):
    """
    record label
    :param dictionary_name:
    :param label: string
    :param label2ids: should be a dictionary of (string,defaultdict)
    :return:
    """
    label2ids[dictionary_name + "_count"]['value'] += 1
    label2ids[dictionary_name][label] = label2ids[dictionary_name+"_count"]['value']

def read_datafile(path):
    word = []
    sentences = []
    # label2ids = {'surface_form': dd(int),
    #              'morph_token': dd(int)}
    label_classes = ['surface_form', 'morph_token']
    label2ids = {label: dd(int) for label in label_classes + [label_class + "_count" for label_class in label_classes]}

    def encode_label(dictionary_name, label):
        """
        record label
        :param dictionary_name:
        :param label: string
        :param label2ids: should be a dictionary of (string,defaultdict)
        :return:
        """
        _encode_label(dictionary_name, label, label2ids)

    with codecs.open(path, mode="r") as f:
        line = f.readline()
        while line:
            tokens = line.split(" ")
            surface_form = tokens[0]
            encode_label('surface_form', surface_form)
            analyses = tokens[1:]
            if surface_form in ["<DOC>", "<TITLE>", "</DOC>", "</TITLE>"]:
                pass # do nothing, skip line
            elif surface_form == "<S>":
                sentence = [] # prepare the sentence variable
            elif surface_form == "</S>":
                # record the sentence
                assert len(sentence) > 0, "sentence with length 0?"
                roots = []
                affixes = []
                for w in sentence:
                    w_roots = []
                    w_affixes = []
                    for analysis in w[1:]:
                        w_roots.append(analysis[0])
                        w_affixes.append(analysis[1:])
                    roots.append(w_roots)
                    affixes.append(w_affixes)
                processed_sentence = {'sentence_length': len(sentence),
                             'surface_forms': [w[0] for w in sentence],
                             'roots': roots,
                             'affixes': affixes}
                # print processed_sentence
                sentences.append(processed_sentence)
            else:
                # this is a legit surface form, extract morph. analyses
                word.append(surface_form)
                for morph_analysis in analyses:
                    morph_tokens = morph_analysis.split("+")
                    for morph_token in morph_tokens:
                        encode_label('morph_token', morph_token)
                    root = morph_tokens[0]
                    affixes = morph_tokens[1:]
                    word.append([root] + affixes)
                # print word
                sentence.append(word)
                word = []
            line = f.readline()
    return sentences, label2ids

if __name__ == "__main__":
    sentences, label2ids = read_datafile("sample.data")
    print sentences
    print label2ids
