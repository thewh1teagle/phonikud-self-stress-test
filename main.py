"""
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx -O phonikud-1.0.int8.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.onnx -O tts-model.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json -O tts-model.config.json
wget https://raw.githubusercontent.com/oyd11/hebrew-text-data/refs/heads/main/hebrew_homographs_stress_minimal_pairs_phrases_llm1.csv -O llm1.csv
wget https://github.com/thewh1teagle/nakdimon-onnx/releases/download/v0.1.0/nakdimon.onnx
"""

import pandas as pd
import phonikud.lexicon
from phonikud_onnx import Phonikud
import phonikud
from tqdm import tqdm
import re
from nakdimon_onnx import Nakdimon


tqdm.pandas()

model_model = Phonikud("phonikud-1.0.int8.onnx") # phonikud_model.add_diacritics will return diacritics
nakdimon_model = Nakdimon("nakdimon.onnx") # nadkimon_modal.compute will return diacritics


def remove_diacritics(text):
    return re.sub(r"[\u0590-\u05c7]", "", text)

def remove_enhanced_diacritics(text):
    # Convert set to regex character class, escaping special characters
    chars_to_remove = ''.join(char for char in phonikud.lexicon.SET_ENHANCED_DIACRITICS)
    pattern = f'[{chars_to_remove}]'
    return re.sub(pattern, "", text)

# Read both CSV files
llm1_df = pd.read_csv("llm1.csv")
saspeech_df = pd.read_csv("saspeech_10.csv", header=None, names=['id', 'vocalized_text', 'phonetic'])

# Process llm1 data
llm1_data = llm1_df[['id', 'phrase']].copy()
llm1_data.columns = ['id', 'unvocalized']
llm1_data['id'] = 'llm1_' + llm1_data['id'].astype(str)

# Process saspeech data - remove diacritics from vocalized text
saspeech_data = saspeech_df[['id', 'vocalized_text']].copy()
saspeech_data['unvocalized'] = saspeech_data['vocalized_text'].apply(remove_diacritics)
saspeech_data = saspeech_data[['id', 'unvocalized']]

# Combine both datasets
combined_df = pd.concat([llm1_data, saspeech_data], ignore_index=True)

# Add vocalized versions using both models
combined_df['vocalized_phonikud_enhanced'] = combined_df['unvocalized'].progress_apply(lambda x: model_model.add_diacritics(x))
# now just vocalized_phonikud, by removing enhanced diacritics
combined_df['vocalized_phonikud'] = combined_df['vocalized_phonikud_enhanced'].progress_apply(lambda x: remove_enhanced_diacritics(x))

combined_df['vocalized_nakdimon'] = combined_df['unvocalized'].progress_apply(lambda x: nakdimon_model.compute(x))

# Save to stress_test_sentences.csv
combined_df.to_csv("stress_test_sentences.csv", index=False)
