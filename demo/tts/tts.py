from transformers import pipeline

# 创建TTS Pipeline对象
pipe = pipeline("text-to-speech")
# 输入文本内容
text = "Hello, my dog is cooler than you!"
# 生成语音
result = pipe(text)
# 打印生成的音频信息和采样率
print(result["sampling_rate"])
print(result["audio"].shape)

# 保存生成的音频文件（可选）
import scipy.io.wavfile
scipy.io.wavfile.write("output.wav", rate=result["sampling_rate"], data=result["audio"])

if __name__ == '__main__':
    pass