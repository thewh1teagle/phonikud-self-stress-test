"""
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/saspeech_automatic_stts2-light_epoch_00010.pth
git clone https://github.com/thewh1teagle/StyleTTS2-lite -b hebrew2
cd StyleTTS2-lite
uv sync
uv run ../generate_audio.py
uv add numpy==2.0.0
uv pip install -e ./StyleTTS2
"""
import sys
import torch
import numpy as np
from pathlib import Path
from functools import lru_cache

# Add parent directory to path to import StyleTTS2
root_dir = Path(__file__).parent / 'StyleTTS2-lite'
sys.path.append(str(root_dir))
from inference import StyleTTS2


class TextToSpeech:
    def __init__(self, config_path, models_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config_path = config_path
        self.models_path = models_path
        self.model = StyleTTS2(config_path, models_path).eval().to(self.device)
    
    @lru_cache(maxsize=128)
    def get_styles(self, speaker_path, speed, denoise, avg_style):
        """Get styles from speaker audio with LRU caching"""
        speaker = {
            "path": speaker_path,
            "speed": speed
        }
        with torch.no_grad():
            return self.model.get_styles(speaker, denoise, avg_style)
    
    def _create(self, phonemes, styles, stabilize=True, alpha=18):
        """Generate audio from phonemes and styles"""
        with torch.no_grad():
            audio = self.model.generate(phonemes, styles, stabilize, alpha)
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            return audio
    
    def create(self, phonemes, speaker_path, speed=0.82, denoise=0.2, avg_style=True, stabilize=True, alpha=18):
        """Complete synthesis pipeline from phonemes to audio with cached styles"""
        # Use cached style extraction
        styles = self.get_styles(speaker_path, speed, denoise, avg_style)
        audio = self._create(phonemes, styles, stabilize, alpha)
        return audio, 24000


_model = None

config_path = str(Path("StyleTTS2-lite") / "Configs" / "config.yaml")
models_path = 'saspeech_automatic_stts2-light_epoch_00010.pth'
ref_audio_path = "StyleTTS2-lite/Demo/Audio/10_michael.wav"

def create(phonemes: str, speaker_path = ref_audio_path):
    global _model
    if _model is None:
        _model = TextToSpeech(config_path, models_path)
    return _model.create(phonemes, speaker_path)



