#!/usr/bin/env bash

cat $1 | awk '{ for (i=2; i <= NF; i++) { n_elems = split($i, arr, "+"); printf "%s\t%s\t", $1, arr[1]; for (j = 2; j <= n_elems; j++) { printf "+%s", arr[j]; }; printf "\n";  }; print "";}'