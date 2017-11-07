#!/usr/bin/env bash

style=${2:-tur_style}

if [[ $style == "tur_style" ]]; then
  cat $1 | awk '/^$/ { print substr(sentence, 2) "\n"; sentence = "" } !/^$/ { sentence = sentence " " $1; }'
else
  cat $1 | awk '/^<\/S>/ { print substr(sentence, 2) "\n"; sentence = "" } !/$<\/S/ { if ($0 ~ /^(<\/?S|<\/?TITLE|<\/?DOC)/) {} else {delimiter = " "; sentence = sentence delimiter $1;} }'
fi