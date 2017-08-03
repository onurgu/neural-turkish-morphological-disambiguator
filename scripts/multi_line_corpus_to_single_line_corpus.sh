#!/usr/bin/env bash

cat $1 | awk '/^$/ { print sentence; sentence = "" } !/^$/ { if (length(sentence) == 0) { sentence = sentence $1; }; for (i = 2; i <= NF; i++ ) { if ( i % 2 == 0) { sentence = sentence " " $i; } else { sentence = sentence $i; } } }'