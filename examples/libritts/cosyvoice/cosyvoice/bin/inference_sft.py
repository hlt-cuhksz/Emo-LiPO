from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import torch
import torchaudio
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.common import IDX_TO_EMO_MAP
import json


def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--emotion_data', required=True, help='emotion data file')
    parser.add_argument('--tts_text', required=True, help='tts input file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--model_dir', required=True, help='model folder')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    cosyvoice = CosyVoice(args.model_dir)

    with open(args.tts_text) as f:
        tts_data = json.load(f)

    with open(args.emotion_data) as f:
        emotion_data = f.readlines()
    emotion_data_dict = {}
    for line in emotion_data:
        key, value = line.split(' ')
        emotion_data_dict[key] = value.strip()

    os.makedirs(args.result_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(tts_data)):
            utts = batch
            tts_text = tts_data[batch][0]
            emotion_id_str = emotion_data_dict[batch]
            spk = utts.split('_')[0]
            instruct_text = IDX_TO_EMO_MAP[emotion_id_str] + '.<endofprompt>' + tts_text
            tts_fn = os.path.join(args.result_dir, '{}.wav'.format(utts))
            for i, j in enumerate(cosyvoice.inference_sft(instruct_text, spk, stream=False)):
                torchaudio.save(tts_fn, j['tts_speech'], sample_rate=22050)


if __name__ == '__main__':
    main()
