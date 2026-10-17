#!/bin/bash
# Draw the 4,000,000 tokenised sentence pairs used for GIZA++ alignment.
# The shuffle is made deterministic with a fixed random source ("yes 42"), which is
# what "seed 42" refers to. Needs GNU coreutils (the exact sample can differ across versions).
#
# Input : <DATA_ROOT>/parallel.{es,eu}.tok   (from tokenize_parallel.py)
# Output: src/giza/parallel_sub.{es,eu}.tok
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${THESIS_DATA:-$HERE/../../data}"

cd "$HERE"
paste "$DATA_ROOT/parallel.es.tok" "$DATA_ROOT/parallel.eu.tok" \
  | shuf --random-source=<(yes 42) \
  | head -n 4000000 \
  | awk -F'\t' '{print $1 > "parallel_sub.es.tok"; print $2 > "parallel_sub.eu.tok"}'

wc -l parallel_sub.es.tok parallel_sub.eu.tok
