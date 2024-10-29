from pydub import AudioSegment


def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio = audio.set_channels(1)  # 转换为单声道
    audio = audio.set_frame_rate(16000)  # 设置采样率为 16kHz
    audio.export(wav_file, format="wav")


convert_mp3_to_wav('output.mp3', 'output.wav')

