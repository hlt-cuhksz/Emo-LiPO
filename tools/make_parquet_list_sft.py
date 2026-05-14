#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import time
import torch
from collections import defaultdict
import random

random.seed(42)

emo_to_idx_map = {
    'Neutral': 0, 
    'Slightly-Angry': 11, 
    'Moderately-Angry': 12, 
    'Extremely-Angry': 13, 
    'Slightly-Happy': 21, 
    'Moderately-Happy': 22, 
    'Extremely-Happy': 23,
    'Slightly-Sad': 31, 
    'Moderately-Sad': 32, 
    'Extremely-Sad': 33,
    'Slightly-Surprised': 41, 
    'Moderately-Surprised': 42, 
    'Extremely-Surprised': 43,
}

idx_to_emo_map = {
    '0': 'Neutral', 
    '11': 'Slightly Angry',
    '12': 'Moderately Angry', 
    '13': 'Extremely Angry', 
    '21': 'Slightly Happy',
    '22': 'Moderately Happy', 
    '23': 'Extremely Happy',
    '31': 'Slightly Sad',
    '32': 'Moderately Sad', 
    '33': 'Extremely Sad',
    '41': 'Slightly Surprised',
    '42': 'Moderately Surprised', 
    '43': 'Extremely Surprised',
}
def emo_to_name(emo):
    if isinstance(emo, int) or (isinstance(emo, str) and emo.isdigit()):
        return idx_to_emo_map[str(emo)]
    return emo

def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    new_utt_list, new_wav_list, new_data_list = [], [], []
    new_text_list, new_spk_list = [], []
    new_uttembedding_list, new_spkembedding_list = [], []
    new_speech_token_list = []

    error_list = []
    for utt in tqdm(utt_list):
        try:
            wav_path = utt2wav[utt]
            with open(wav_path, 'rb') as f:
                audio_bin = f.read()
            txt = utt2text[utt]
            spk = utt2spk[utt]
            emo = utt2emo[utt]
            emo_name = emo_to_name(emo)

            utt_emb = utt2embedding[utt]
            spk_emb = spk2embedding[spk]
            speech_tok = utt2speech_token[utt]

            new_utt_list.append(utt)
            new_wav_list.append(wav_path)
            new_data_list.append(audio_bin)
            new_text_list.append(f"{emo_name}.<endofprompt>{txt}")
            new_spk_list.append(spk)
            new_uttembedding_list.append(utt_emb)
            new_spkembedding_list.append(spk_emb)
            new_speech_token_list.append(speech_tok)
        except Exception as e:
            error_list.append((utt, str(e)))
            logging.exception(f"fail utt={utt}")

    df = pd.DataFrame()
    df['utt'] = new_utt_list
    df['wav'] = new_wav_list
    df['audio_data'] = new_data_list
    df['text'] = new_text_list
    df['spk'] = new_spk_list
    df['utt_embedding'] = new_uttembedding_list
    df['spk_embedding'] = new_spkembedding_list
    df['speech_token'] = new_speech_token_list

    df.to_parquet(parquet_file)

    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in new_utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(new_spk_list))}, f, ensure_ascii=False, indent=2)

    logging.info('spend time {}s'.format(time.time() - start_time))
    logging.info(f"error num={len(error_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet', type=int, default=1000)
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--des_dir', type=str)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk, utt2emo = {}, {}, {}, {}
    with open(f'{args.src_dir}/wav.scp') as f:
        for l in f:
            l = l.strip().split()
            utt2wav[l[0]] = l[1]
    with open(f'{args.src_dir}/text') as f:
        for l in f:
            l = l.strip().split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open(f'{args.src_dir}/utt2spk') as f:
        for l in f:
            l = l.strip().split()
            utt2spk[l[0]] = l[1]
    with open(f'{args.src_dir}/utt2emo') as f:
        for l in f:
            l = l.strip().split()
            utt2emo[l[0]] = l[1]

    text2utt = defaultdict(list)
    for k, v in utt2text.items():
        text2utt[v].append(k)

    utt2embedding = torch.load(f'{args.src_dir}/utt2embedding.pt')
    spk2embedding = torch.load(f'{args.src_dir}/spk2embedding.pt')
    utt2speech_token = torch.load(f'{args.src_dir}/utt2speech_token.pt')
    utts = list(utt2wav.keys())

    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, f'parquet_{i:09d}.parquet')
        utt2parquet_file = os.path.join(args.des_dir, f'utt2parquet_{i:09d}.json')
        spk2parquet_file = os.path.join(args.des_dir, f'spk2parquet_{i:09d}.json')
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        job(utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file)

    with open(f'{args.des_dir}/data.list', 'w', encoding='utf8') as f1, \
         open(f'{args.des_dir}/utt2data.list', 'w', encoding='utf8') as f2, \
         open(f'{args.des_dir}/spk2data.list', 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
