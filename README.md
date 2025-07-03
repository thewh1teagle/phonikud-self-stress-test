
phonikud_vocalized_epoch=5159-step=3145742



https://huggingface.co/thewh1teagle/phonikud-experiments/tree/main/comparison/audio/phonikud_vocalized_epoch%3D5159-step%3D3145742


mkdir -p ./web/audio

for f in ./roboshaul/*.wav; do
  filename=$(basename "$f" .wav)
  ffmpeg -i "$f" -c:a aac -b:a 128k "./web/audio/${filename}.m4a"
done



with saspeech

```console
uv venv -p3.10
brew install mecab
git clone https://github.com/shenberg/TTS
uv pip install -e TTS
uv pip install gdown
uvx gdown 1dExa0AZqmyjz8rSZz1noyQY9aF7dR8ew
uvx gdown 1eK1XR_ZwuUy4yWh80nui-q5PBifJsYfy
uvx gdown 1XdmRRHjZ_eZOFKoAQgQ8wivrLDJnNDkh
uvx gdown 1An6cTCYkxXWhagIJe3NGkoP8n2CQWQ-3
mkdir tts_model
mkdir hifigan_model
mv saspeech_nikud_7350.pth tts_model/
mv config_overflow.json tts_model/
mv checkpoint_500000.pth hifigan_model/
mv config_hifigan.json hifigan_model/

CUDA_VISIBLE_DEVICES= uv run tts --text "שָׁלוֹם וּבְרָכָה נִפָּרֵד בְּשִׂמְחָה מִמֻּמֵּן" \
        --model_path tts_model/saspeech_nikud_7350.pth \
        --config_path tts_model/config_overflow.json \
        --vocoder_path hifigan_model/checkpoint_500000.pth \
        --vocoder_config_path hifigan_model/config_hifigan.json \
        --out_path test.wav
```