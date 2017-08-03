from __future__ import print_function
import argparse
import codecs
import subprocess
import tempfile
from collections import defaultdict as dd
import os
import pickle
import sys

from asciitree import LeftAligned, Traversal, draw_tree
from collections import OrderedDict as OD

from nltk import RegexpTokenizer


class TreeNode():

    def __init__(self, parent, node_label=""):

        self.parent = parent
        self.node_label = node_label
        self.end_node = 0
        self.children = OD()

    def __contains__(self, item):
        return item in self.children

    def __getitem__(self, item):
        return self.children[item]

    def new_child(self, new_child_node):
        self.children[new_child_node.node_label]  = new_child_node

    def print_children(self):
        print(self.node_label, ": ", self.children.keys())

    def print_children_recursive(self, od=None, depth=0):

        od = OD()

        if len(self.children) == 0:
            return {}
        else:
            for child_label, child in self.children.items():
                od[child_label] = child.print_children_recursive(depth=depth+1)
            if depth != 0:
                return od
            else:
                return OD({'ROOT': od})

        # for child_label, child in self.children.items():
        #     if len(child.children) > 0:
        #         child.print_children()
        #     child.print_children_recursive()

def insert_into_tree(root, path_as_node_labels):
    current_node = root
    for iter_idx, next_node_label in enumerate(path_as_node_labels):
        if next_node_label in current_node:
            current_node = current_node[next_node_label]
        else:
            current_node.new_child(TreeNode(current_node, next_node_label))
            current_node = current_node[next_node_label]
    current_node.end_node += 1

def count_tagsets(f, delimiter="\t", gold_analysis_in_the_first_position=False, verbose=False):
    tagsets_dict = dd(int)
    root_and_analysis_cooccurence = {}
    surface_form_and_gold_analysis_cooccurence = {}

    ambiguity_scores = []

    def record(key, key2, d):
        if key in d:
            d[key][key2] += 1
        else:
            d[key] = dd(int)
            d[key2] = 1

    def record_root_and_analysis_cooccurence(root, analysis):
        record(root, analysis, root_and_analysis_cooccurence)

    def record_surface_form_and_gold_analysis_cooccurence(surface_form, analysis):
        record(surface_form, analysis, surface_form_and_gold_analysis_cooccurence)

    current_tagset = []
    current_roots = []
    analyses_idx = 0

    sentence_length = 0

    line = f.readline()
    # print line
    while line:
        line = line.strip()
        tokens = line.split(delimiter)
        # print tokens
        if len(tokens) == 3:
            if gold_analysis_in_the_first_position and analyses_idx == 0:
                record_surface_form_and_gold_analysis_cooccurence(tokens[0], tokens[2])
            if analyses_idx == 0 and verbose:
                print("SURFACE FORM: %s" % tokens[0])
            current_tagset += [tokens[2]]
            current_roots += [tokens[1]]
            record_root_and_analysis_cooccurence(tokens[1], tokens[2])
            analyses_idx += 1

            if tokens[0] in ["<S>", "<DOC>", "<TITLE>", "</DOC>", "</TITLE>"]:
                sentence_length = 0
                current_product_of_ambiguities = 1
            elif tokens[0] == "</S>":
                ambiguity_score = current_product_of_ambiguities / float(sentence_length) if sentence_length != 0 else 0.0
                ambiguity_scores.append([ambiguity_score, sentence_length])

        elif len(tokens) == 1:
            # tagset ended
            if len(current_tagset) > 0:
                tree_root = TreeNode(None, "ROOT")
                root_to_anonymized_root = {root: ("X%d" % (idx+1)) for idx, root in enumerate(sorted(set(current_roots)))}
                sorted_tagset = sorted(zip([root_to_anonymized_root[root] for root in current_roots], current_tagset), key=lambda x: x[1])
                tagsets_dict["\n".join([x + y for x, y in sorted_tagset])] += 1

                current_product_of_ambiguities *= len(current_tagset)

                # trees
                for tagset_as_seq in [(x + y).split("+") for x, y in sorted_tagset]:
                    insert_into_tree(tree_root, tagset_as_seq)

                if verbose:
                    unanonymized_sorted_tagset = sorted(
                        zip(current_roots,
                            current_tagset), key=lambda x: x[1])
                    print(unanonymized_sorted_tagset)
                    print(sorted_tagset)
                    tr = LeftAligned()
                    print(tr(tree_root.print_children_recursive()))

                # clear
                current_tagset = []
                current_roots = []
                analyses_idx = 0

                sentence_length += 1

        elif len(tokens) == 2:
            # <DOC> or <TITLE> OR <S> OR </S>
            pass
        line = f.readline()
    return tagsets_dict, root_and_analysis_cooccurence, surface_form_and_gold_analysis_cooccurence, ambiguity_scores


def conll2003tosingleline():
    pass


import operator
from functools import reduce


def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def tokenize(line):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    return tokenizer.tokenize(line)


def create_single_word_single_line_format(string_output):
    lines = string_output.split("\n")
    result = "<S> <S>+BSTag\n"
    current_single_line = ""
    subline_idx = 0
    for line in lines:
        if line != "":
            tokens = line.split("\t")
            if subline_idx == 0:
                current_single_line += tokens[0]
                current_single_line += " " + tokens[1] + tokens[2]
            else:
                current_single_line += " " + tokens[1] + tokens[2]
            subline_idx += 1
        else:
            result += current_single_line + "\n"
            subline_idx = 0
            current_single_line = ""
    result = result[:-1]
    result += "</S> </S>+ESTag\n"
    return result

def get_morph_analyzes(line):
    """

    :param line: 
    :return: 
    """
    if type(line) == unicode:
        tokens = tokenize(line)
    else:
        tokens = tokenize(line.decode("utf8"))
    # print tokens
    fd, f_path = tempfile.mkstemp()
    with open(f_path, "w") as f:
        for token in tokens:
            f.write(token.encode("iso-8859-9") + "\n")
    os.close(fd)
    with codecs.open(f_path, "r", encoding="iso-8859-9") as f:
        string_output = subprocess.check_output(["./bin/lookup", "-latin1", "-f",
                                                 "tfeatures.scr"], stdin=f, cwd="./tools/tr-tagger")
    return string_output

def calculate_ambiguity_score_of_a_sentence(line):
    morph_analyzes_output = get_morph_analyzes(line)
    single_lined_morph_analyzes_output = \
        create_single_word_single_line_format(morph_analyzes_output)
    counts = [(len(x.split(" "))-1) for x in single_lined_morph_analyzes_output.split("\n")[1:-1]]
    return prod(counts)/float(len(single_lined_morph_analyzes_output.split("\n"))-2-1), \
           single_lined_morph_analyzes_output, counts


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True, choices=["generate_corpus_statistics", "gui"])
    parser.add_argument("--gold_data", type=bool, default=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()

    if args.command == "generate_corpus_statistics":
        f = sys.stdin

        tagsets_dict, root_and_analysis_cooccurence, surface_form_and_gold_analysis_cooccurence, ambiguity_scores = \
            count_tagsets(f, gold_analysis_in_the_first_position=args.gold_data, verbose=(True if args.verbose == "1" else False))

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        for filename, obj in [["tagsets_dict", tagsets_dict],
                         ["root_and_analysis_cooccurence", root_and_analysis_cooccurence],
                         ["surface_form_and_gold_analysis_cooccurence", surface_form_and_gold_analysis_cooccurence],
                         ["ambiguity_scores", ambiguity_scores]]:
            with open(os.path.join(args.output_dir, filename+".dat"), "w") as out_f:
                pickle.dump(obj, out_f)
        f.close()
    elif args.command == "gui":

        from PyQt4 import QtGui
        import main_form

        class ExampleApp(QtGui.QMainWindow, main_form.Ui_MainWindow):
            def __init__(self):
                super(self.__class__, self).__init__()
                self.setupUi(self)  # This is defined in design.py file automatically

                self.pushButton.clicked.connect(self.calculate_ambiguity)

                self.plainTextEdit.setPlainText("Ali ata bak")

            def calculate_ambiguity(self):
                single_line_free_text_sentence = str(self.plainTextEdit.toPlainText())
                ambiguity_score, single_lined_morph_analyzes_output, counts = calculate_ambiguity_score_of_a_sentence(single_line_free_text_sentence)

                self.label.setText("%lf" % ambiguity_score)
                self.plainTextEdit_2.setPlainText(single_lined_morph_analyzes_output)
                from PyQt4.QtCore import QStringList
                self.listWidget.clear()
                self.listWidget.addItems(QStringList([" ".join(x) for x in zip(["N/A"] + [str(y) for y in counts[:-1]] + ["N/A"],
                                                                               single_lined_morph_analyzes_output.split("\n")[:-1])]))

        app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
        form = ExampleApp()  # We set the form to be our ExampleApp (design)
        form.show()  # Show the form
        app.exec_()  # and execute the app