#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# bash script to calculate sentence embeddings for arbitrary
# text file

if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit 1
fi


lang=$1

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

rm dev.nl.bpe
rm dev.nl.lbl
rm dev.nl.tok
rm dev.nl.txt
rm dev.nl.txt.enc

python3 embed_token.py \
    --encoder ${encoder} \
    --token-lang ${lang} \
    --bpe-codes ${bpe_codes} \
    --verbose
