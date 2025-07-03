"""
wget https://huggingface.co/thewh1teagle/phonikud-experiments-checkpoints/resolve/main/phonikud_enhanced/epoch=5159-step=3145742.onnx -O phonikud_enhanced.onnx
wget https://huggingface.co/thewh1teagle/phonikud-experiments-checkpoints/resolve/main/phonikud_vocalized/epoch=5159-step=3145742.onnx -O phonikud_vocalized.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json -O tts-model.config.json
"""

import pandas as pd
import phonikud
from tqdm import tqdm
from phonikud_tts import Piper
from pathlib import Path
import soundfile as sf
from mms import Mms

wav_dir = Path('./wav')
wav_dir.mkdir(exist_ok=True)

modal1_path = Path('./phonikud_enhanced.onnx')
modal2_path = Path('./phonikud_vocalized.onnx')
piper1 = Piper(modal1_path, config_path=Path('./tts-model.config.json'))
piper2 = Piper(modal2_path, config_path=Path('./tts-model.config.json'))

# Initialize MMS model
mms = Mms("facebook/mms-tts-heb")

df = pd.read_csv('web/stress_test_sentences.csv')
df = df.dropna()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating audio"):
    id_val = row['id']
    
    # # Generate phonikud_enhanced audio
    # if pd.notna(row['vocalized_phonikud_enhanced']):
    #     phonemes = phonikud.phonemize(row['vocalized_phonikud_enhanced'])
    #     samples,sample_rate = piper1.create(phonemes, is_phonemes=True)
    #     sf.write(wav_dir / f'vocalized_phonikud_enhanced_{id_val}.wav', samples, sample_rate)
    
    # # Generate phonikud audio
    # if pd.notna(row['vocalized_phonikud']):
    #     phonemes = phonikud.phonemize(row['vocalized_phonikud'])
    #     samples,sample_rate = piper2.create(phonemes, is_phonemes=True)
    #     sf.write(wav_dir / f'vocalized_phonikud_{id_val}.wav', samples, sample_rate)
    
    # Generate MMS audio using nakdimon text
    if pd.notna(row.get('vocalized_nakdimon', row.get('nakdimon', row.get('text', '')))):
        nakdimon_text = row.get('vocalized_nakdimon', row.get('nakdimon', row.get('text', '')))
        samples, sample_rate = mms.generate(nakdimon_text)
        sf.write(wav_dir / f'mms_nakdimon_{id_val}.wav', samples, sample_rate) 