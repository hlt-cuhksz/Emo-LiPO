import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from tqdm import tqdm
import time

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct_lipo', "rule_with_neutral_5_index")
print(cosyvoice.list_available_spks())

prompt_mapping = {
    "neutral_0": "Neutral.",
    "happy_1": "Slightly Happy.",
    "happy_2": "Moderately Happy.",
    "happy_3": "Extremely Happy.",
    "sad_1": "Slightly Sad.",
    "sad_2": "Moderately Sad.",
    "sad_3": "Extremely Sad.",
    "angry_1": "Slightly Angry.",
    "angry_2": "Moderately Angry.",
    "angry_3": "Extremely Angry.",
    "surprised_1": "Slightly Surprised.",
    "surprised_2": "Moderately Surprised.",
    "surprised_3": "Extremely Surprised."
}

with open('text', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Processing text lines"):
        utt, text = line.split(' ', 1)
        speaker4prompt = utt.split('_')
        speaker = speaker4prompt[0]
        prompt_identifier = '_'.join(speaker4prompt[2:])
        prompt = prompt_mapping[prompt_identifier]
        print(f'Processing {utt} with prompt: {prompt} and speaker: {speaker}')
        for j in cosyvoice.inference_instruct(text, speaker, prompt, stream=False):
            torchaudio.save(f'result/Audio/{utt}.wav', j['tts_speech'], cosyvoice.sample_rate)