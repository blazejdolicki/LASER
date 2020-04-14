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
mldir="data/cls-acl10-unprocessed"	# raw texts of MLdoc
edir="embed"	# normalized texts and embeddings
# languages=('en' 'de' 'fr' 'jp')
languages=('en' 'de')
# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

###################################################################
#
# Download raw data and convert to txt files (to make them compatible with processing)
#
###################################################################

echo "Start time:"
TZ=CET date
echo ""

# if data wasn't downloaded before
if [ ! -d "data" ] ; then
    echo "Downloading CLS dataset"
    DATA_DIR='data'
    mkdir ${DATA_DIR}
    url="https://zenodo.org/record/3251672/files/cls-acl10-unprocessed.tar.gz"
    wget -O data/cls-acl10-unprocessed.tar.gz ${url}

    # check if download failed
    if [ $? -ne 0 ]; then
        echo "Download failed. Check if the data set is still hosted at ${url}"
        exit
    fi
    echo "Unpacking tar file"
    tar -C ${DATA_DIR} -xzf ${DATA_DIR}/cls-acl10-unprocessed.tar.gz 
    echo "Converting xml to txt files"
    python convert_to_txt.py
else
    echo "Dataset already exists. Skipping download. "
fi


###################################################################
#
# Extract files with labels and texts from the MLdoc corpus
#
###################################################################


ExtractCLS () {
  # input file name
  ifname=$1
  # output file name
  ofname=$2
  # language
  lang=$3

  if [ ! -f ${ofname}.lbl.${lang} ] ; then
    echo " - extract labels from ${ifname}"
    cut -d$'\t' -f1 ${ifname} > ${ofname}.lbl.${lang}
  else
    echo "Labels from ${ifname} already extracted"
  fi
  if [ ! -f ${ofname}.txt.${lang} ] ; then
    echo " - extract texts from ${ifname}"
    cut -d$'\t' -f2 ${ifname} \
      > ${ofname}.txt.${lang}
  else
    echo "Texts from ${ifname} already extracted"
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
echo -e "\nExtracting CLS data"

for part in "train" "dev" "test" ; do
  for l in ${languages[@]} ; do
    ExtractCLS "${mldir}/${l}/books/${part}.txt" ${edir}/${part} ${l}
  done
done

MECAB="${LASER}/tools-external/mecab"
export LD_LIBRARY_PATH="${MECAB}/lib:${LD_LIBRARY_PATH}"
python3 cls.py --data_dir ${edir} --lang ${languages[@]} --bpe_codes ${bpe_codes} --encoder ${encoder}

# CLS classifier parameters
nb_cl=2
Ns=(100 200 500)
lrs=(0.01 0.001)
wd=0.0
nhids=("10 8" "256 64")
drops=(0.1 0.2 0.3)
seed=1
bsizes=(12 18)

echo -e "\nTraining CLS classifier (log files in ${edir})"

hp_tuning=true
if [ "$hp_tuning" = true ] ; then
  for ltrn in "en" ; do
  # TODO uncomment later
  # for ltrn in ${languages[@]} ; do
    ldev=${ltrn}
    echo " - train on ${ltrn}, dev on ${ldev}"
    for N in ${Ns[@]} ; do
      for lr in ${lrs[@]} ; do
        for i in ${!nhids[@]} ; do
          nhid=${nhids[i]}
          for drop in ${drops[@]} ; do
            for bsize in ${bsizes[@]} ; do
              lf="${edir}/cls.${ltrn}-${ldev}_N_${N}_lr_${lr}_nhid_${i}_drop_${drop}_bsize_${bsize}.log"
              echo "Starting ${lf}"
              # TODO uncomment later
              # if [ ! -f ${lf} ] ; then
                python3 ${LASER}/source/sent_classif.py \
                  --gpu 0 --base-dir ${edir} \
                  --train train.enc.${ltrn} \
                  --train-labels train.lbl.${ltrn} \
                  --dev dev.enc.${ldev} \
                  --dev-labels dev.lbl.${ldev} \
                  --test test.enc \
                  --test-labels test.lbl \
                  --nb-classes ${nb_cl} \
                  --nhid ${nhid[@]} --dropout ${drop} --bsize ${bsize} \
                  --seed ${seed} --lr ${lr} --wdecay ${wd} --nepoch ${N} \
                  --lang ${languages[@]} \
                  > ${lf}
              # fi
            done
          done
        done
      done
    done
  done
else
  for ltrn in "en" ; do
  # for ltrn in ${languages[@]} ; do
    ldev=${ltrn}
    lf="${edir}/cls.${ltrn}-${ldev}.log"
    echo " - train on ${ltrn}, dev on ${ldev}"
    if [ ! -f ${lf} ] ; then
      python3 ${LASER}/source/sent_classif.py \
        --gpu 0 --base-dir ${edir} \
        --train train.enc.${ltrn} \
        --train-labels train.lbl.${ltrn} \
        --dev dev.enc.${ldev} \
        --dev-labels dev.lbl.${ldev} \
        --test test.enc \
        --test-labels test.lbl \
        --nb-classes ${nb_cl} \
        --nhid ${nhid[@]} --dropout ${drop} --bsize ${bsize} \
        --seed ${seed} --lr ${lr} --wdecay ${wd} --nepoch ${N} \
        --lang ${languages[@]} \
        > ${lf}
    fi
  done
fi



# # display results
# echo -e "\nAccuracy matrix:"
# echo -n "Train "
# for l1 in ${languages[@]} ; do
#   printf "    %2s " ${l1}
# done
# echo ""
# for l1 in ${languages[@]} ; do
#   lf="${edir}/mldoc.${l1}-${l1}.log"
#   echo -n " ${l1}:  "
#   for l2 in ${languages[@]} ; do
#     grep "Test lang ${l2}" $lf | sed -e 's/%//' | awk '{printf("  %5.2f", $10)}'
#   done
#   echo ""
# done
