#!/usr/bin/env bash

cat $1 | awk '{ if ($1 != "<DOC>" || $1 != "<TITLE> || $1 != "</DOC>" || $1 != "</TITLE>") { print; if ($1 == "<S>") { in_sentence = 1; } else if ($1 != "</S>") { if (in_sentence == 1) { printf "%s ", $1; }} } }'