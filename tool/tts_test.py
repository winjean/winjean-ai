from edge_tts import Communicate
import asyncio

# text: 要合成的文本
# voice: 语音种类，可以是英文或中文，例如：
# en-US-JennyNeural zh-CN-YunxiaNeural zh-CN-XiaoxiaoNeural


async def edge_tts_test(text, voice="zh-CN-YunxiaNeural"):
    communicate = Communicate(
        text=text,
        voice=voice,
        # rate="medium",
        # volume="medium",
    )
    await communicate.save("output.mp3")


# asyncio.run(edge_tts_test("你好，这是一个TTS测试."))

# 获取当前事件循环
loop = asyncio.get_event_loop()

try:
    # 运行事件循环，直到 my_coroutine 完成
    loop.run_until_complete(edge_tts_test("你好，这是一个TTS测试."))
finally:
    # 关闭事件循环
    loop.close()

