from modelscope.hub.snapshot_download import snapshot_download

cache_dir = "E:/model/text-to-speech/"
model_dir = snapshot_download('damo/speech_sambert-hifigan_tts_zh-cn_16k', cache_dir=cache_dir, revision='master')




if __name__ == '__main__':
    pass