import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
# import multiprocessing
import time
import torch
from collections import defaultdict, OrderedDict
import random
import csv
import base64

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

# 反查：把索引数字（如 '11'）映射回标准名称（如 'Slightly-Angry'）
idx_to_name = {str(v): k for k, v in emo_to_idx_map.items()}
valid_emo_names = set(emo_to_idx_map.keys())

# 5种不同的情感强度排列顺序
EMOTION_ORDER_PATTERNS = [
    ['Extremely', 'Moderately', 'Slightly', 'Neutral'],
    ['Moderately', 'Slightly', 'Extremely', 'Neutral'],
    ['Slightly', 'Moderately', 'Neutral', 'Extremely'],
]

# 四个基础情感，每个需要齐备 5 档强度
BASE_EMOTIONS = ['Angry', 'Happy', 'Sad', 'Surprised']

# 额外3种情感，强度随机，每种7列（无text）
EXTRA_EMOTIONS = ['Angry', 'Happy', 'Sad', 'Surprised']  # 从4种中随机选3种
EXTRA_INTENSITIES = ['Slightly', 'Moderately', 'Extremely']  # 随机强度

# 运行时可选：将服务器路径前缀替换为本地前缀
PATH_PREFIX_MAP = []  # 形如 [(server_prefix, local_prefix), ...]
DIR_NAME_MAP = {}     # 形如 { 'Slightly-Angry': 'angry_1', ... }

def normalize_emo_label(raw):
    """
    将 utt2emo 的原始值（可能是数字字符串、数字或名称）统一到标准名称：
    'Neutral' 或 '{Degree}-{BaseEmotion}'。
    """
    if raw is None:
        return None
    s = str(raw)
    if s in idx_to_name:
        return idx_to_name[s]
    if s in valid_emo_names:
        return s
    return None

def format_prompt(emo_name: str, text: str) -> str:
    """根据 emo 名称查找其索引，再用 idx_to_emo_map 的提示语拼接文本。"""
    try:
        emo_idx = emo_to_idx_map.get(emo_name, 0)
        prompt = idx_to_emo_map.get(str(emo_idx), emo_name)
    except Exception:
        prompt = emo_name
    return f"{prompt}.<endofprompt>{text}" # 因为prompt自身有.符号，所以需要加<endofprompt>

def resolve_path(p: str) -> str:
    """根据 PATH_PREFIX_MAP 将服务器绝对路径替换为本地路径。"""
    norm = p.replace('\\', '/')
    mapped = norm
    # 前缀替换
    for old, new in PATH_PREFIX_MAP:
        old_norm = old.replace('\\', '/')
        if mapped.startswith(old_norm):
            suffix = mapped[len(old_norm):]
            mapped = (new.replace('\\', '/') + suffix).replace('//', '/')
            break
    # 目录名替换（逐段替换匹配的目录名）
    parts = mapped.split('/')
    parts = [DIR_NAME_MAP.get(seg, seg) for seg in parts]
    mapped = '/'.join(parts)
    return mapped

def job(group_keys, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()

    # 准备 8 + 7*7 = 57 列
    df_cols = {}
    
    # 第1部分：8列（包含text）
    df_cols.update({
        'utt': [], 'wav': [], 'audio_data': [], 'text': [], 'spk': [],
        'utt_embedding': [], 'spk_embedding': [], 'speech_token': [],
    })
    
    # 第2-8部分：每部分7列（不包含text）
    for i in range(2, 8):  # utt2 到 utt8
        df_cols.update({
            f'utt{i}': [], f'wav{i}': [], f'audio_data{i}': [], f'spk{i}': [],
            f'utt_embedding{i}': [], f'spk_embedding{i}': [], f'speech_token{i}': [],
        })

    used_utts = set()
    used_spks = set()
    missing_counter = 0

    for (spk, text, base_emo) in tqdm(group_keys):
        cand_utts = text2utt.get(text, [])
        cand_utts = [u for u in cand_utts if utt2spk.get(u) == spk]

        # 构建所有可能的情感标签
        all_emotion_names = ['Neutral']
        for degree in ['Slightly', 'Moderately', 'Extremely']:
            all_emotion_names.append(f'{degree}-{base_emo}')

        # 收集所有需要的情感样本
        emotion_bucket = {name: None for name in all_emotion_names}
        for u in cand_utts:
            emo_name = normalize_emo_label(utt2emo.get(u))
            if emo_name in emotion_bucket and emotion_bucket[emo_name] is None:
                emotion_bucket[emo_name] = u
            if all(emotion_bucket.values()):
                break

        if not all(emotion_bucket.values()):
            missing_counter += 1
            continue

        # 为每种排列模式生成一行数据
        for pattern_idx, emotion_order in enumerate(EMOTION_ORDER_PATTERNS):
            # 为每一行选择3种不同的基础情感作为额外数据（排除当前base_emo）
            available_emotions = [e for e in BASE_EMOTIONS if e != base_emo]
            extra_emotions = random.sample(available_emotions, min(3, len(available_emotions)))
            extra_bucket = {}
            
            for i, extra_emo in enumerate(extra_emotions):
                # 随机选择强度
                random_intensity = random.choice(EXTRA_INTENSITIES)
                extra_emo_name = f'{random_intensity}-{extra_emo}'
                
                # 寻找匹配的样本
                for u in cand_utts:
                    emo_name = normalize_emo_label(utt2emo.get(u))
                    if emo_name == extra_emo_name:
                        extra_bucket[f'extra{i+1}'] = u
                        break
                if f'extra{i+1}' not in extra_bucket:
                    # 如果找不到精确匹配，找该情感的任何强度
                    for u in cand_utts:
                        emo_name = normalize_emo_label(utt2emo.get(u))
                        if emo_name and emo_name.startswith(extra_emo):
                            extra_bucket[f'extra{i+1}'] = u
                            break
            # 第1部分：8列（包含text）- 使用第一个情感
            first_emo = emotion_order[0]
            if first_emo == 'Neutral':
                first_emo_name = 'Neutral'
            else:
                first_emo_name = f'{first_emo}-{base_emo}'
            
            first_utt = emotion_bucket[first_emo_name]
            first_wav = resolve_path(utt2wav[first_utt])
            first_data = open(first_wav, 'rb').read()
            
            df_cols['utt'].append(first_utt)
            df_cols['wav'].append(first_wav)
            df_cols['audio_data'].append(first_data)
            df_cols['text'].append(format_prompt(first_emo_name, text))
            df_cols['spk'].append(spk)
            df_cols['utt_embedding'].append(utt2embedding[first_utt])
            df_cols['spk_embedding'].append(spk2embedding[spk])
            df_cols['speech_token'].append(utt2speech_token[first_utt])

            used_utts.add(first_utt)
            used_spks.add(spk)

            # 第2-6部分：7列（不包含text）- 使用其余4种情感
            for i, degree in enumerate(emotion_order[1:], 2):
                if degree == 'Neutral':
                    emo_name = 'Neutral'
                else:
                    emo_name = f'{degree}-{base_emo}'
                
                u = emotion_bucket[emo_name]
                w = resolve_path(utt2wav[u])
                data = open(w, 'rb').read()

                df_cols[f'utt{i}'].append(u)
                df_cols[f'wav{i}'].append(w)
                df_cols[f'audio_data{i}'].append(data)
                df_cols[f'spk{i}'].append(spk)
                df_cols[f'utt_embedding{i}'].append(utt2embedding[u])
                df_cols[f'spk_embedding{i}'].append(spk2embedding[spk])
                df_cols[f'speech_token{i}'].append(utt2speech_token[u])

                used_utts.add(u)

            # 第6部分：使用额外情感1（不同的基础情感）
            extra_key = 'extra1'
            if extra_key in extra_bucket and extra_bucket[extra_key] is not None:
                u = extra_bucket[extra_key]
                w = resolve_path(utt2wav[u])
                data = open(w, 'rb').read()
                
                df_cols['utt5'].append(u)
                df_cols['wav5'].append(w)
                df_cols['audio_data5'].append(data)
                df_cols['spk5'].append(spk)
                df_cols['utt_embedding5'].append(utt2embedding[u])
                df_cols['spk_embedding5'].append(spk2embedding[spk])
                df_cols['speech_token5'].append(utt2speech_token[u])
                
                used_utts.add(u)
            else:
                # 如果找不到匹配的样本，填充空值
                df_cols['utt5'].append('')
                df_cols['wav5'].append('')
                df_cols['audio_data5'].append(b'')
                df_cols['spk5'].append('')
                df_cols['utt_embedding5'].append(None)
                df_cols['spk_embedding5'].append(None)
                df_cols['speech_token5'].append(None)

            # 第7-8部分：使用额外情感2-3（不同的基础情感）
            for i in range(2):
                extra_key = f'extra{i+2}'  # extra2, extra3
                col_idx = i + 6  # utt7, utt8
                
                if extra_key in extra_bucket and extra_bucket[extra_key] is not None:
                    u = extra_bucket[extra_key]
                    w = resolve_path(utt2wav[u])
                    data = open(w, 'rb').read()
                    
                    df_cols[f'utt{col_idx}'].append(u)
                    df_cols[f'wav{col_idx}'].append(w)
                    df_cols[f'audio_data{col_idx}'].append(data)
                    df_cols[f'spk{col_idx}'].append(spk)
                    df_cols[f'utt_embedding{col_idx}'].append(utt2embedding[u])
                    df_cols[f'spk_embedding{col_idx}'].append(spk2embedding[spk])
                    df_cols[f'speech_token{col_idx}'].append(utt2speech_token[u])
                    
                    used_utts.add(u)
                else:
                    # 如果找不到匹配的样本，填充空值
                    df_cols[f'utt{col_idx}'].append('')
                    df_cols[f'wav{col_idx}'].append('')
                    df_cols[f'audio_data{col_idx}'].append(b'')
                    df_cols[f'spk{col_idx}'].append('')
                    df_cols[f'utt_embedding{col_idx}'].append(None)
                    df_cols[f'spk_embedding{col_idx}'].append(None)
                    df_cols[f'speech_token{col_idx}'].append(None)

    if missing_counter > 0:
        print(f'[Info] skipped groups (missing any of 5 degrees): {missing_counter}')

    # 保存到 parquet 与映射
    df = pd.DataFrame(df_cols)
    df.to_parquet(parquet_file)

    with open(utt2parquet_file, 'w', encoding='utf-8') as f:
        json.dump({k: parquet_file for k in sorted(used_utts)}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w', encoding='utf-8') as f:
        json.dump({k: parquet_file for k in sorted(used_spks)}, f, ensure_ascii=False, indent=2)

    logging.info('spend time {}'.format(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num (spk,text,base_emo) groups per parquet (each group generates 5 rows with different emotion orders)')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--prefix_map',
                        type=str,
                        nargs='*',
                        default=[],
                        help='server-to-local path prefix mappings, format: OLD|NEW (can pass multiple)')
    parser.add_argument('--dir_map',
                        type=str,
                        nargs='*',
                        default=[],
                        help='server-to-local directory name mappings, format: OLDNAME|NEWNAME (can pass multiple)')
    parser.add_argument('--write_preview_csv',
                        action='store_true',
                        help='also write a light-weight CSV (no binary/tensors) for easy viewing')
    parser.add_argument('--csv_include_audio_base64',
                        action='store_true',
                        help='include audio_data columns in preview CSV as base64 text (large)')
    parser.add_argument('--csv_audio_base64_maxchars',
                        type=int,
                        default=64,
                        help='when including audio base64 in CSV, truncate to at most N characters')
    args = parser.parse_args()
    # 解析前缀映射
    if args.prefix_map:
        for item in args.prefix_map:
            if '|' in item:
                old, new = item.split('|', 1)
                if old:
                    PATH_PREFIX_MAP.append((old, new))
    if args.dir_map:
        for item in args.dir_map:
            if '|' in item:
                old, new = item.split('|', 1)
                if old:
                    DIR_NAME_MAP[old] = new

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
    # 构造 (spk, text, base_emo) 组合作为一行单位
    pairs = set()
    for u, t in utt2text.items():
        s = utt2spk[u]
        for base_emo in BASE_EMOTIONS:
            pairs.add((s, t, base_emo))
    pairs = sorted(list(pairs))

    # 确保输出目录存在（在写 parquet 之前）
    os.makedirs(args.des_dir, exist_ok=True)

    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    start = 0
    i = 1
    total = len(pairs)
    while start < total:
        stop = min(start + args.num_utts_per_parquet, total)
        batch_keys = pairs[start:stop]

        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.parquet'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)

        job(batch_keys, parquet_file, utt2parquet_file, spk2parquet_file)

        # 可选：写出轻量 CSV 便于查看
        if args.write_preview_csv:
            try:
                import pandas as _pd
                # 选择可读字段，按新的列结构排列；是否包含 audio_data 由开关控制
                include_audio = bool(args.csv_include_audio_base64)
                preview_cols = []
                
                # 第1部分：8列（包含text）
                preview_cols.extend(['utt','wav'])
                if include_audio:
                    preview_cols.append('audio_data')
                preview_cols.extend(['text','spk','utt_embedding','spk_embedding','speech_token'])
                
                # 第2-8部分：每部分7列（不包含text）
                for i in range(2, 8):
                    preview_cols.extend([f'utt{i}', f'wav{i}'])
                    if include_audio:
                        preview_cols.append(f'audio_data{i}')
                    preview_cols.extend([f'spk{i}', f'utt_embedding{i}', f'spk_embedding{i}', f'speech_token{i}'])
                # 读取刚写的 parquet 再挑列，避免在内存里重复维护两份
                df_full = _pd.read_parquet(parquet_file)
                have_cols = [c for c in preview_cols if c in df_full.columns]
                df_prev = df_full[have_cols].copy()
                # 可选：将音频二进制转为 base64 文本
                if args.csv_include_audio_base64:
                    audio_cols = [c for c in have_cols if c.endswith('audio_data')]
                    maxchars = max(0, int(args.csv_audio_base64_maxchars))
                    for col in audio_cols:
                        def _enc(x):
                            if isinstance(x, (bytes, bytearray)):
                                s = base64.b64encode(x).decode('ascii')
                                if maxchars and len(s) > maxchars:
                                    return s[:maxchars] + '...'
                                return s
                            return '' if x is None else str(x)
                        df_prev[col] = df_prev[col].apply(_enc)
                csv_path = parquet_file.replace('.parquet', '_preview.csv')
                df_prev.to_csv(csv_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
            except Exception as e:
                logging.warning('fail to write preview csv: {}'.format(e))

        start = stop
        i += 1

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
