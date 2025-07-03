
phonikud_vocalized_epoch=5159-step=3145742



https://huggingface.co/thewh1teagle/phonikud-experiments/tree/main/comparison/audio/phonikud_vocalized_epoch%3D5159-step%3D3145742


mkdir -p ./web/audio

for f in ./wav/*.wav; do
  filename=$(basename "$f" .wav)
  ffmpeg -i "$f" -c:a aac -b:a 128k "./web/audio/${filename}.m4a"
done


with saspeech use

https://gist.github.com/thewh1teagle/04252cbcf5cf32f395f5fc0498f8fee2