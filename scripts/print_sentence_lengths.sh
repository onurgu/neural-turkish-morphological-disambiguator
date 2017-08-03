#!/usr/bin/env bash

cat $1 | awk '/^<S>/ { sentence_count = 0; } !/^(<\/?S)|(<DOC)|(<\/?TITLE)/ { sentence_count += 1} /^<\/S>/ { print sentence_count }'