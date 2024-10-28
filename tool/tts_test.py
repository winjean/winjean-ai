from edge_tts import Communicate
import asyncio

""" 
text: 要合成的文本
voice: 语音种类，可以是英文或中文，例如：
"""


async def edge_tts_test():
    communicate = Communicate(
        text="你好，这是一个TTS测试.",
        # voice="en-US-JennyNeural",
        voice="zh-CN-XiaoxiaoNeural",
        # rate="medium",
        # volume="medium",
        # output_file="output.mp3"
    )
    await communicate.save("output.mp3")


asyncio.run(edge_tts_test())

