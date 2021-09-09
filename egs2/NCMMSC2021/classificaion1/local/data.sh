#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    # local/data fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') $*"
}
SECONDS=0
stage=0
stop_stage=2
rawdata_dir=
max_segment_duration=12
overlap_duration=3
datadir=/blob/tsst/users/v-weiwang1/espnet/egs2/NCMMSC2021/classification/data
audio_format=wav
nj=8
fs=16k

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${NCMMSC2021}" ]; then
    log "Fill the value of 'NCMMSC2021' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    for dset in AD HC MCI; do
        mkdir -p ${datadir}/local/data/${dset}
        find ${NCMMSC2021}/${dset}/ -name *.wav  > ${datadir}/local/data/${dset}/wav.flist
        awk 'BEGIN{FS="[/.]"} {print $(NF-1), $0}' ${datadir}/local/data/${dset}/wav.flist  > ${datadir}/local/data/${dset}/wav.scp
        cut -d" " -f1 ${datadir}/local/data/${dset}/wav.scp > ${datadir}/local/data/${dset}/uttids
        paste -d" " ${datadir}/local/data/${dset}/uttids <(cut -d"_" -f1-3 ${datadir}/local/data/${dset}/uttids) > ${datadir}/local/data/${dset}/utt2spk
        utils/utt2spk_to_spk2utt.pl ${datadir}/local/data/${dset}/utt2spk > ${datadir}/local/data/${dset}/spk2utt
        cut -d" " -f1 ${datadir}/local/data/${dset}/spk2utt > ${datadir}/local/data/${dset}/spkids
        paste -d" " ${datadir}/local/data/${dset}/spkids <(cut -f2 -d"_" ${datadir}/local/data/${dset}/spkids) | awk '{print $1,tolower($2)}' > ${datadir}/local/data/${dset}/spk2gender
        utils/data/get_reco2dur.sh ${datadir}/local/data/${dset}

        mkdir -p ${datadir}/${dset}/
        for f in wav.scp spk2utt utt2spk spk2gender reco2dur; do
            cp ${datadir}/local/data/${dset}/$f ${datadir}/${dset}/
        done
        utils/fix_data_dir.sh ${datadir}/${dset}
    done
    awk '{print $1,0}' ${datadir}/HC/wav.scp > ${datadir}/HC/utt2cat
    awk '{print $1,1}' ${datadir}/MCI/wav.scp > ${datadir}/MCI/utt2cat
    awk '{print $1,2}' ${datadir}/AD/wav.scp > ${datadir}/AD/utt2cat
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: make train, dev, test set (8:1:1)"

    # for dset in AD HC MCI; do
    for dset in AD HC MCI; do
        utils/subset_data_dir_tr_cv.sh --cv-spk-percent 20 ${datadir}/${dset}/ ${datadir}/local/data/${dset}_tr ${datadir}/local/data/${dset}_cv > /dev/null
        utils/subset_data_dir_tr_cv.sh --cv-spk-percent 50 ${datadir}/local/data/${dset}_cv/ ${datadir}/local/data/${dset}_dev ${datadir}/local/data/${dset}_test > /dev/null
    done

    utils/combine_data.sh --extra-files "utt2cat" ${datadir}/train ${datadir}/local/data/AD_tr ${datadir}/local/data/HC_tr ${datadir}/local/data/MCI_tr
    utils/combine_data.sh --extra-files "utt2cat" ${datadir}/dev ${datadir}/local/data/AD_dev ${datadir}/local/data/HC_dev ${datadir}/local/data/MCI_dev
    utils/combine_data.sh --extra-files "utt2cat" ${datadir}/test ${datadir}/local/data/AD_test ${datadir}/local/data/HC_test ${datadir}/local/data/MCI_test

    for dset in train dev test; do
        utils/fix_data_dir.sh ${datadir}/${dset}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: make uniform segments"

    # for dset in AD HC MCI; do
    for dset in train dev test; do
        utils/data/get_utt2dur.sh ${datadir}/${dset}
        utils/data/get_segments_for_data.sh ${datadir}/${dset} > ${datadir}/${dset}/segments
	  	utils/data/get_uniform_subsegments.py \
			--max-segment-duration=$max_segment_duration \
			--overlap-duration=$overlap_duration \
			--max-remaining-duration=$(perl -e "print $max_segment_duration / 2.0") \
			${datadir}/${dset}/segments > ${datadir}/${dset}/uniform_sub_segments
        utils/data/subsegment_data_dir.sh ${datadir}/${dset} ${datadir}/${dset}/uniform_sub_segments ${datadir}/${dset}_seg
        utils/fix_data_dir.sh --utt-extra-files "utt2cat" ${datadir}/local/data/${dset}_seg
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
