import deepspeech
import wave
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# deepspeech sst
# wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
# wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# 加载预训练模型
model_path = 'deepspeech-0.9.3-models.pbmm'
scorer_path = 'deepspeech-0.9.3-models.scorer'

model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)


def read_wav_file(filename):
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
    return buffer, rate


audio_file = 'output.wav'
audio_buffer, sample_rate = read_wav_file(audio_file)

text = model.stt(audio_buffer)
print(f"Recognized text: {text}")
