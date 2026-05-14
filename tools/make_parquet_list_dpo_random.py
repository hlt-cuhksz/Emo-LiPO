#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
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

def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(utt2wav[utt], 'rb').read()
        data_list.append(data)
    
    wav_list = [utt2wav[utt] for utt in utt_list]
    text_list = [utt2text[utt] for utt in utt_list]
    spk_list = [utt2spk[utt] for utt in utt_list]
    uttembedding_list = [utt2embedding[utt] for utt in utt_list]
    spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list]
    speech_token_list = [utt2speech_token[utt] for utt in utt_list]
    emo_list = [utt2emo[utt] for utt in utt_list]
    
    new_utt_list = []
    new_data_list = []
    new_wav_list = []
    new_text_list = []
    new_spk_list = []
    new_uttembedding_list = []
    new_spkembedding_list = []
    new_speech_token_list = []
    new_emo_list = []
    paired_utt_list = []
    error_list = []
    for idx, text in enumerate(text_list):
        try:
            temp_utt_list = [item for item in text2utt[text] if item != utt_list[idx] and spk_list[idx] == utt2spk[item]]
            assert len(temp_utt_list) > 0, f"{idx} -- {text} --- {spk_list[idx]} --- {utt_list[idx]} -- {text2utt[text]}"
            for rejected_utt in temp_utt_list:
                if utt2emo[rejected_utt] != emo_list[idx]:
                    paired_utt_list.append(rejected_utt)
                    new_utt_list.append(utt_list[idx])
                    new_data_list.append(data_list[idx])
                    new_wav_list.append(wav_list[idx])
                    new_text_list.append(text_list[idx])
                    new_spk_list.append(spk_list[idx])
                    new_uttembedding_list.append(uttembedding_list[idx])
                    new_spkembedding_list.append(spkembedding_list[idx])
                    new_speech_token_list.append(speech_token_list[idx])
                    new_emo_list.append(emo_list[idx])
                else:
                    continue
        except:
            error_list.append(idx)
            
    print(len(paired_utt_list))
    print(len(error_list))
    
    paired_data_list = []
    for utt in tqdm(paired_utt_list):
        data = open(utt2wav[utt], 'rb').read()
        paired_data_list.append(data)
    
    paired_wav_list = [utt2wav[utt] for utt in paired_utt_list]
    # paired_text_list = [utt2text[utt] for utt in paired_utt_list]
    # for x, y in zip(text_list, paired_text_list):
    #     assert x == y
    
    new_text_list = [idx_to_emo_map[emo] + f".<endofprompt>{t}" for t, emo in zip(new_text_list, new_emo_list)]
    paired_spk_list = [utt2spk[utt] for utt in paired_utt_list] 
    paired_uttembedding_list = [utt2embedding[utt] for utt in paired_utt_list]
    paired_spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in paired_utt_list]
    paired_speech_token_list = [utt2speech_token[utt] for utt in paired_utt_list]
    
    # utt_list = [item for idx, item in enumerate(utt_list) if idx not in error_list]
    # wav_list = [item for idx, item in enumerate(wav_list) if idx not in error_list]
    # data_list = [item for idx, item in enumerate(data_list) if idx not in error_list]
    # text_list = [item for idx, item in enumerate(text_list) if idx not in error_list]
    # spk_list = [item for idx, item in enumerate(spk_list) if idx not in error_list]
    # uttembedding_list = [item for idx, item in enumerate(uttembedding_list) if idx not in error_list]
    # spkembedding_list = [item for idx, item in enumerate(spkembedding_list) if idx not in error_list]
    # speech_token_list = [item for idx, item in enumerate(speech_token_list) if idx not in error_list]
    

    # 保存到parquet,utt2parquet_file,spk2parquet_file
    df = pd.DataFrame()
    df['utt'] = new_utt_list
    df['wav'] = new_wav_list
    df['audio_data'] = new_data_list
    df['text'] = new_text_list
    df['spk'] = new_spk_list
    df['utt_embedding'] = new_uttembedding_list
    df['spk_embedding'] = new_spkembedding_list
    df['speech_token'] = new_speech_token_list
    
    df['rejected_utt'] = paired_utt_list
    df['rejected_wav'] = paired_wav_list
    df['rejected_audio_data'] = paired_data_list
    df['rejected_spk'] = paired_spk_list
    df['rejected_utt_embedding'] = paired_uttembedding_list
    df['rejected_spk_embedding'] = paired_spkembedding_list
    df['rejected_speech_token'] = paired_speech_token_list
    
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk, utt2emo = {}, {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
            
    with open('{}/utt2emo'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2emo[l[0]] = l[1]
            
    text2utt = defaultdict(list)
    for k, v in utt2text.items():
        text2utt[v].append(k)
            
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))
    utts = list(utt2wav.keys())

    # Using process pool to speedup
    # pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.parquet'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        job(utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file)
    #     pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    # pool.close()
    # pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
