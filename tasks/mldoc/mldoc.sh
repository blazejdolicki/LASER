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
# bash script to calculate sentence embeddings for the MLDoc corpus,
# train and evaluate the classifier

if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

# general config
mldir="MLDoc"	# raw texts of MLdoc
edir="embed"	# normalized texts and embeddings
languages=('en' 'de' 'es' 'fr' 'it' 'ja' 'ru' 'zh')

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

edir="embed"

###################################################################
#
# Extract files with labels and texts from the MLdoc corpus
#
###################################################################

```
  ExtractMLdoc:

  Take the original MLDOC document. It seems to be delimited by tabs where in each line the second value is the label
  and the third is the corresponding news article. It takes the labels, replaces some values and saves them in a separate file.
  Afterwards, it takes the articles, replaces some values and saves in a separate file.

```

ExtractMLdoc () {
  # input file name
  ifname=$1
  # output file name
  ofname=$2
  # language
  lang=$3
  if [ ! -f ${ifname}.${lang} ] ; then
    echo "Please install the MLDoc corpus first"
    exit
  fi

  if [ ! -f ${ofname}.lbl.${lang} ] ; then
    echo " - extract labels from ${ifname}.${lang}"
    cut -d'	' -f1 ${ifname}.${lang} \
      | sed -e 's/C/0/' -e 's/E/1/' -e 's/G/2/' -e 's/M/3/' \
      > ${ofname}.lbl.${lang}
  fi
  if [ ! -f ${ofname}.txt.${lang} ] ; then
    echo " - extract texts from ${ifname}.${lang}"
    # remove text which is not useful for classification 
    '''
      `cut` command cuts a section from each line and writes it to a file https://shapeshed.com/unix-cut/ 
      -d specifies the delimiter (seems to be a tab in this case)
      -f after splitting by delimeter we have a list, this parameter selects the nth element from that list

      `sed` command is can various uses, but here it just replaces one pattern with another https://www.geeksforgeeks.org/sed-command-in-linux-unix-with-examples/
      The "s" specifies the substitution operation. The "g" specifies the sed command to replace all occurences of the string in the line.
      -e just binds multiple commands together
      So here basically "Co ." is replaced with "Co.", "Inc ." with "Inc." and "the "Reteurs Limited.." with nothing (its removed).

      At the end everything is saved into a file.
    '''
    cut -d'	' -f2 ${ifname}.${lang} \
     | sed -e 's/ Co \./ Co./g' -e s'/ Inc \. / Inc. /g' \
           -e 's/([cC]) Reuters Limited 199[0-9]\.//g' \
      > ${ofname}.txt.${lang}
  fi
}


###################################################################
#
# Create all files
#
###################################################################

# create output directories
for d in ${edir} ; do
  mkdir -p ${d}
done

# Embed all data
echo -e "\nExtracting MLDoc data"
#ExtractMLdoc ${mldir}/mldoc.train1000 ${edir}/mldoc.train1000 "en"
for part in "mldoc.train1000" "mldoc.dev" "mldoc.test" ; do
  for l in ${languages[@]} ; do
    ExtractMLdoc ${mldir}/${part} ${edir}/${part} ${l}
  done
done

MECAB="${LASER}/tools-external/mecab"
export LD_LIBRARY_PATH="${MECAB}/lib:${LD_LIBRARY_PATH}"
python3 mldoc.py --data_dir ${edir} --lang ${languages[@]} --bpe_codes ${bpe_codes} --encoder ${encoder}

# MLDoc classifier parameters
nb_cl=4
N=500
lr=0.001
wd=0.0
nhid="10 8"
drop=0.2
seed=1
bsize=12

echo -e "\nTraining MLDoc classifier (log files in ${edir})"
#for ltrn in "en" ; do
for ltrn in ${languages[@]} ; do
  ldev=${ltrn}
  lf="${edir}/mldoc.${ltrn}-${ldev}.log"
  echo " - train on ${ltrn}, dev on ${ldev}"
  if [ ! -f ${lf} ] ; then
    python3 ${LASER}/source/sent_classif.py \
      --gpu 0 --base-dir ${edir} \
      --train mldoc.train1000.enc.${ltrn} \
      --train-labels mldoc.train1000.lbl.${ltrn} \
      --dev mldoc.dev.enc.${ldev} \
      --dev-labels mldoc.dev.lbl.${ldev} \
      --test mldoc.test.enc \
      --test-labels mldoc.test.lbl \
      --nb-classes ${nb_cl} \
      --nhid ${nhid[@]} --dropout ${drop} --bsize ${bsize} \
      --seed ${seed} --lr ${lr} --wdecay ${wd} --nepoch ${N} \
      --lang ${languages[@]} \
      > ${lf}
  fi
done

# display results
echo -e "\nAccuracy matrix:"
echo -n "Train "
for l1 in ${languages[@]} ; do
  printf "    %2s " ${l1}
done
echo ""
for l1 in ${languages[@]} ; do
  lf="${edir}/mldoc.${l1}-${l1}.log"
  echo -n " ${l1}:  "
  for l2 in ${languages[@]} ; do
    grep "Test lang ${l2}" $lf | sed -e 's/%//' | awk '{printf("  %5.2f", $10)}'
  done
  echo ""
done
