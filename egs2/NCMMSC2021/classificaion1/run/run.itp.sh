#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_seg"
valid_set="dev_seg"
test_sets="test_seg"

# asr_config=conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml
classification_config=conf/train_classification_s3prl_tdnn_hubert_base.yaml
inference_config=conf/decode_asr.yaml

dumpdir=/blob/tsst/users/v-weiwang1/espnet/egs2/NCMMSC2021/classification1/dump
expdir=/blob/tsst/users/v-weiwang1/espnet/egs2/NCMMSC2021/classification1/exp
datadir=/blob/tsst/users/v-weiwang1/espnet/egs2/NCMMSC2021/classification1/data
nltk_data_dir=/blob/tsst/users/v-weiwang1/nltk_data

./classification.itp.sh \
    --ngpu 8 \
    --max_wav_duration 30 \
    --classification_config "${classification_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir  "${dumpdir}" \
    --nltk_data_dir "${nltk_data_dir}" \
    --expdir "${expdir}" \
    --datadir "${datadir}" "$@"
