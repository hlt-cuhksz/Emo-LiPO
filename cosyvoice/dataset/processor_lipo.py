import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

torchaudio.set_audio_backend('soundfile')

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

SUFFIXES = ['', '2', '3', '4', '5', '6', '7']


def parquet_opener(data, mode='train', tts_data={}):
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            df = pq.read_table(url).to_pandas()
            for i in range(len(df)):
                if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                    continue
                sample.update(dict(df.loc[i]))
                if mode == 'train':
                    yield {**sample}
                else:
                    for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                        yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data, max_length=10240, min_length=10, token_max_length=200,
           token_min_length=1, min_output_input_ratio=0.0005,
           max_output_input_ratio=1, mode='train'):
    for sample in data:
        # Load audio for all candidates
        valid = True
        for s in SUFFIXES:
            sample[f'speech{s}'], sample[f'sample_rate{s}'] = torchaudio.load(BytesIO(sample[f'audio_data{s}']))
            del sample[f'audio_data{s}']
            num_frames = sample[f'speech{s}'].size(1) / sample[f'sample_rate{s}'] * 100
            if num_frames < min_length or num_frames > max_length:
                valid = False
                break
            if len(sample[f'speech_token{s}']) == 0:
                valid = False
                break

        if not valid:
            continue
        if len(sample['text_token']) < token_min_length:
            continue
        if len(sample['text_token']) > token_max_length:
            continue

        num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100
        if num_frames != 0:
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    for sample in data:
        valid = True
        for s in SUFFIXES:
            assert f'speech{s}' in sample
            assert f'sample_rate{s}' in sample
            sr = sample[f'sample_rate{s}']
            if sr != resample_rate:
                if sr < min_sample_rate:
                    valid = False
                    break
                sample[f'sample_rate{s}'] = resample_rate
                sample[f'speech{s}'] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate)(sample[f'speech{s}'])
            max_val = sample[f'speech{s}'].abs().max()
            if max_val > 1:
                sample[f'speech{s}'] /= max_val
        if valid:
            yield sample


def compute_fbank(data, feat_extractor, mode='train'):
    for sample in data:
        for s in SUFFIXES:
            assert f'sample_rate{s}' in sample
            assert f'speech{s}' in sample
            waveform = sample[f'speech{s}']
            if waveform.size(0) == 2:
                waveform = torch.mean(waveform, dim=0).unsqueeze(0)
            sample[f'speech_feat{s}'] = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
            del sample[f'speech{s}']
        yield sample


def parse_embedding(data, normalize, mode='train'):
    for sample in data:
        for s in SUFFIXES:
            sample[f'utt_embedding{s}'] = torch.tensor(sample[f'utt_embedding{s}'], dtype=torch.float32)
            sample[f'spk_embedding{s}'] = torch.tensor(sample[f'spk_embedding{s}'], dtype=torch.float32)
            if normalize:
                sample[f'utt_embedding{s}'] = F.normalize(sample[f'utt_embedding{s}'], dim=0)
                sample[f'spk_embedding{s}'] = F.normalize(sample[f'spk_embedding{s}'], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode='train'):
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train'):
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample], dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)

        batch = {}

        # Text (shared across all candidates)
        batch['text'] = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        batch['text_token_len'] = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        batch['text_token'] = pad_sequence(text_token, batch_first=True, padding_value=0)

        # Per-candidate fields
        for s in SUFFIXES:
            utt_key = 'utt' if s == '' else f'utt{s}'
            batch[f'utts{s}' if s == '' else f'utts{s}'] = [sample[i][utt_key] for i in order]
            # Use 'utts' for the first candidate to match expected key
            if s == '':
                batch['utts'] = [sample[i]['utt'] for i in order]

            speech_token = [torch.tensor(sample[i][f'speech_token{s}']) for i in order]
            batch[f'speech_token_len{s}'] = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
            batch[f'speech_token{s}'] = pad_sequence(speech_token, batch_first=True, padding_value=0)

            speech_feat = [sample[i][f'speech_feat{s}'] for i in order]
            batch[f'speech_feat_len{s}'] = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
            batch[f'speech_feat{s}'] = pad_sequence(speech_feat, batch_first=True, padding_value=0)

            batch[f'utt_embedding{s}'] = torch.stack([sample[i][f'utt_embedding{s}'] for i in order], dim=0)
            batch[f'spk_embedding{s}'] = torch.stack([sample[i][f'spk_embedding{s}'] for i in order], dim=0)

        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text, 'tts_index': tts_index,
                          'tts_text_token': tts_text_token, 'tts_text_token_len': tts_text_token_len})

        emb_key = 'spk_embedding' if use_spk_embedding else 'utt_embedding'
        for s in SUFFIXES:
            batch[f'embedding{s}'] = batch[f'{emb_key}{s}']

        yield batch
