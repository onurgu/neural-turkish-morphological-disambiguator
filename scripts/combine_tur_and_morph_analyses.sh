#!/usr/bin/env bash

for label in train test ; do
	echo $label;
	cd tools/tr-tagger ;
	bash ../../scripts/multi_line_sentence_to_single_line_sentence.sh ../../data/tr.${label} tur_style | awk '!/^$/ { for (i = 1; i <= NF; i++) { print $i; }} /^$/ { print "</S>"; }' | iconv -f utf-8 -t iso-8859-9 | ./bin/lookup -latin1 -f tfeatures.scr | iconv -f iso-8859-9 -t utf-8 > ../../data/tr.${label}.with_raw_morph_tags ; cd -;
	cat data/tr.${label}.with_raw_morph_tags | awk 'BEGIN { in_word = 0 } !/(^$)/ { if (in_word == 1) { morph_analyses = morph_analyses " " $2 $3; } else { in_word = 1; surface_form = $1; morph_analyses = $2 $3; } } /^$/ { print	 morph_analyses; in_word = 0;}' | awk '{ if ($0 ~ /^<\/S/) { print ""; } else { print } } ' > data/tr.${label}.with_morph_tags_single_line ;
	cat data/tr.${label} | awk '{ print $1, $2, $3 }' | paste -d" " - data/tr.${label}.with_morph_tags_single_line > data/tr.${label}.combined_with_morph_tags ;
	less data/tr.${label}.combined_with_morph_tags | awk '!/^$|^\s+$/ { found = 0; for (i = 4; i <= NF; i++) { if ($2 == $i) { found = 1} }; if (found == 0) { print "NOT FOUND IN ANALYSES: ", $0; } else { found = 0 } } ' | wc -l && wc -l data/tr.${label}.combined_with_morph_tags ;
done