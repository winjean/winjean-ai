import edge_tts
import time
import asyncio

# text: 要合成的文本
# voice: 语音种类，可以是英文或中文，例如：
# en-US-JennyNeural zh-CN-YunxiaNeural zh-CN-XiaoxiaoNeural
# edge-tts --list-voices

audio_path = r"../audio"
output_wav_path = f"{audio_path}/output.wav"


async def edge_tts_test(text, voice="zh-CN-YunxiaNeural"):
    start_time = time.time()
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        # rate="medium",
        # volume="medium",
    )
    await communicate.save(output_wav_path)
    print("[TTS] Edge TTS infer cost:", time.time() - start_time)


if __name__ == "__main__":
    # 获取当前事件循环
    # loop = asyncio.get_event_loop()
    #
    # try:
    #     # 运行事件循环，直到 my_coroutine 完成
    #     loop.run_until_complete(edge_tts_test("上午好，很高兴为您服务."))
    # finally:
    #     # 关闭事件循环
    #     loop.close()
    asyncio.run(edge_tts_test("上午好，很高兴为您服务."))
