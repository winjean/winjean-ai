import pyttsx3
import wave
import struct

"""
windows、linux、mac需要安装不同的语音引擎
"""

def on_start(name):
    print(f"Starting: {name}")

def on_word(name, location, length):
    print(f"Word: {name} at {location} with length {length}")

def on_end(name, completed):
    print(f"Finished: {name}, completed={completed}")

# 初始化 pyttsx3 引擎
engine = pyttsx3.init()

# 获取所有可用的语音
# engine.setProperty('voice', 'zh-cn')
# voices = engine.getProperty('voices')
#
# for voice in voices:
#     print(f"ID: {voice.id}")
#     print(f"Name: {voice.name}")
#     print(f"Languages: {voice.languages}")
#     print(f"Gender: {voice.gender}")
#     print(f"Age: {voice.age}")
#     print("-" * 40)



# 注册事件处理器
engine.connect('started-utterance', on_start)
engine.connect('started-word', on_word)
engine.connect('finished-utterance', on_end)

# 要转换的文本
# engine.say("Hello, World!")
text = "你好，欢迎使用语音合成技术。语音已保存"

# 设置语音属性
# engine.setProperty('rate', 150)  # 语音速度
# engine.setProperty('volume', 1.0)  # 语音音量

# engine.say(text)
# 保存音频文件
output_file = "output.mp3"
# 保存语音到文件
engine.save_to_file(text, output_file)
engine.runAndWait()

print("语音已保存到 output.wav")
