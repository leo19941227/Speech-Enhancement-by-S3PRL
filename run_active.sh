#!/bin/bash

config=$1
noise_dir="../datasets/noise_data/Nonspeech_digits/"

for noise_type in $(ls $noise_dir);
do
    noise_str=$noise_dir"/"$noise_type
    python3 run_downstream.py --name active_p232_$noise_type --expdir result/active/all_noise --ckpt ../S3PRL/result/se7-noise2clean/states-500000.ckpt --ckpt2 ../S3PRL/result/noisy2noise-6layer-3072/states-500000.ckpt --downstream LSTM --dckpt result/active/pretrain/3lstm-l1-v2/states-500000.ckpt --from_rawfeature --config $config --active_sampling --sync_sampler --test_speech "../datasets/speech_data/clean_testset_wav_16k/p232*" --test_noise $noise_str --eval_init
done

