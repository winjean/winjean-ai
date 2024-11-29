#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/6 17:03
@Author  : lxc
@File    : volcengineASR.py
@Desc    :
curl -X POST -H 'Accept: */*' -H 'Authorization: Bearer; gMFujoojCvqipnvQtso4ODVDwpc_lDGl' -H 'Connection: keep-alive' -H 'User-Agent: python-requests/2.22.0' -H 'content-type: application/json' -d '{"url": "http://146.56.226.252:8010/files/1106.WAV", "audio_text": "长沙口味虾，又称“口味虾”或“麻辣小龙虾”，是湖南省长沙市的特色小吃之一，以其独特的麻辣口味和鲜香口感而闻名。这道菜的主要食材是新鲜的小龙虾，经过清洗、去头、去肠等处理后，加入大量的辣椒、花椒和其他香料进行烹饪。口味虾色泽红亮，肉质鲜嫩，辣中带麻，回味无穷，是夏季夜市和聚餐时的热门选择。在长沙，无论是街头小吃摊还是高档餐厅，都能品尝到这道地道的美食。"}' 'https://openspeech.bytedance.com/api/v1/vc/ata/submit?appid=4565900214&caption_type=speech'


 {"id":"2c9fbe22-a9c1-4a96-9bc6-85e53a8b0b4b","code":0,"message":"Success"}



 curl -X GET -H 'Accept: */*' -H 'Authorization: Bearer; gMFujoojCvqipnvQtso4ODVDwpc_lDGl' -H 'Connection: keep-alive' -H 'User-Agent: python-requests/2.22.0' 'https://openspeech.bytedance.com/api/v1/vc/ata/query?appid=4565900214&id=2c9fbe22-a9c1-4a96-9bc6-85e53a8b0b4b'

返回json格式
"""
import os
import subprocess
import requests
import cv2
base_url = 'https://openspeech.bytedance.com/api/v1/vc/ata'
appid = "4565900214"
access_token = "gMFujoojCvqipnvQtso4ODVDwpc_lDGl"
def get_video_ratio(video_file):
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 关闭视频文件
    cap.release()
    return width/height
def get_asr(file, audio_text):
    response = requests.post(
        '{base_url}/submit'.format(base_url=base_url),
        params=dict(
            appid=appid,
            caption_type='speech',
        ),
        files={
            'audio-text': audio_text,
            'data': (os.path.basename(file), open(file, 'rb'), 'audio/wav'),
        },
        headers={
            'Authorization': 'Bearer; {}'.format(access_token)
        }
    )
    print('submit response = {}'.format(response.text))
    assert (response.status_code == 200)
    assert (response.json()['message'] == 'Success')

    job_id = response.json()['id']
    response = requests.get(
        '{base_url}/query'.format(base_url=base_url),
        params=dict(
            appid=appid,
            id=job_id,
        ),
        headers={
            'Authorization': 'Bearer; {}'.format(access_token)
        }
    )
    print('query response = {}'.format(response.json()))
    assert (response.status_code == 200)

    return response.json()["utterances"]

def get_srt(audio_text, ratio, file, srt_path):
    # 存储合并后的字幕内容
    subtitles = []
    # 用于临时存储当前句子的单词和时间戳
    current_sentence = []
    start_time = 0.0
    # 设置最长字幅
    max_length = int(ratio * 20)
    results = get_asr(file, audio_text)
    for segment in results:
        for word_info in segment['words']:
            word = word_info['text']
            start = word_info['start_time']
            end = word_info['end_time']
            if not word:
                continue
            # 检查是否为切分点（逗号或句号）
            if word[-1] in [',', '。', '，']:
                # 合并成一句话并保存
                if current_sentence:
                    current_sentence.append(word[:-1])
                    sentence = ''.join(current_sentence).replace(',', '').replace('。', '')
                    sentence = sentence[:max_length] + "\n" + sentence[max_length:] if "\n" not in sentence and len(
                        sentence) > max_length else sentence
                    subtitles.append((start_time, end, sentence))
                    current_sentence = []  # 清空当前句子
                continue  # 跳过标点符号
            # 如果是第一个单词，记录开始时间
            if not current_sentence:
                start_time = start
            # 添加单词到当前句子
            current_sentence.append(word)
        # 处理段落结束时的剩余句子
        if current_sentence:
            sentence = ''.join(current_sentence).replace(',', '').replace('。', '')
            sentence = sentence[:max_length] + "\n" + sentence[max_length:] if "\n" not in sentence and len(
                sentence) > max_length else sentence
            subtitles.append((start_time, end, sentence))
            current_sentence = []  # 清空当前句子
    # 保存为 SRT 文件格式
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles):
            start = start / 1000
            end = end / 1000
            # 将时间格式转换为 SRT 格式
            start_srt = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int((start % 1) * 1000):03}"
            end_srt = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int((end % 1) * 1000):03}"
            f.write(f"{i + 1}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")
    print("字幕已保存为 output.srt")
    return srt_path




if __name__ == '__main__':
    video_file = "/root/workspace/bgsub_server/4bab948a077b45cca284a77badf3ba7b.mp4"
    audio_text = "长沙口味虾，又称“口味虾”或“麻辣小龙虾”，是湖南省长沙市的特色小吃之一，以其独特的麻辣口味和鲜香口感而闻名。这道菜的主要食材是新鲜的小龙虾，经过清洗、去头、去肠等处理后，加入大量的辣椒、花椒和其他香料进行烹饪。口味虾色泽红亮，肉质鲜嫩，辣中带麻，回味无穷，是夏季夜市和聚餐时的热门选择。在长沙，无论是街头小吃摊还是高档餐厅，都能品尝到这道地道的美食。"
    ratio = get_video_ratio(video_file)
    srt_path = "output.srt"
    get_srt(audio_text, ratio, video_file, srt_path)

    command = f"ffmpeg -i {os.path.abspath(video_file)} -vf subtitles={os.path.abspath(srt_path)} output1.mp4"
    print(f"command:::{command}")
    # 等待命令执行完成并获取输出
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 获取输出和错误信息
    stdout, stderr = process.communicate()
    # 打印输出
    if stdout:
        print(stdout.decode("utf-8").strip())
    if stderr:
        print(stderr.decode("utf-8").strip())
    # 打印返回码
    return_code = process.returncode
    print(f"子进程已结束，返回码：{return_code}")