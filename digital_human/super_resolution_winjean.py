#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2024/11/20 下午2:23
@Author  : winjean
@File    : super_resolution.py
@Software: PyCharm
@Desc:
超分辨率重建
githup地址 https://github.com/TencentARC/GFPGAN
"""


import os
import subprocess
import time
import uuid
import shutil
import logging
import json
import math
import threading

logger = logging.getLogger(__name__)


def gfpgan(task_name: str, video_name: str, audio_name: str, gpu_id: str,
           gfpgan_upscale: int, gfpgan_version: str, sharding_count: int) -> str:
    out_path = f"./tasks/{task_name}/output"
    logger.info("加载GFPGAN模型...")
    logger.info("*" * 50)
    input_audio_path = f"./tasks/{task_name}/{audio_name}"
    input_video_path = f"./tasks/{task_name}/{video_name}"
    unprocessed_frame_path = os.path.join(out_path, "unprocess")
    processed_frame_path = os.path.join(out_path, "process")
    video_list_path = os.path.join(out_path, "video_list")
    os.makedirs(unprocessed_frame_path, exist_ok=True)
    os.makedirs(processed_frame_path, exist_ok=True)
    os.makedirs(video_list_path, exist_ok=True)

    start_time_1 = time.time()
    # 原始视频转换为图片
    video_to_image(input_video_path, unprocessed_frame_path, sharding_count)
    start_time_2 = time.time()
    print("1.原始视频转换为图片完成,耗时：", start_time_2 - start_time_1)

    # 原始图片转换成高清图片
    handle_image_path(gpu_id, unprocessed_frame_path, processed_frame_path, gfpgan_upscale, gfpgan_version)
    start_time_3 = time.time()
    print("2.原始图片转换成高清图片完成,耗时：", start_time_3 - start_time_2)

    # 将图片转换为分段视频
    merge_image_path_to_video(processed_frame_path, video_list_path)
    start_time_4 = time.time()
    print("3.将图片转换为分段视频完成,耗时：", start_time_4 - start_time_3)

    # 合并成一个完整的视频
    concatenate_videos(video_list_path, os.path.join(out_path, "video.mp4"))
    start_time_5 = time.time()
    print("4.合并成一个完整的视频完成,耗时：", start_time_5 - start_time_4)

    merge_video_file = os.path.join(out_path, "video.mp4")
    final_processed_output_video = os.path.join(out_path, "final_video.mp4")
    merge_audio_video(input_audio_path, merge_video_file, final_processed_output_video)
    start_time_6 = time.time()
    print("5.音频和视频合并完成,耗时：", start_time_6 - start_time_5)

    return os.path.abspath(final_processed_output_video)


def super_resolution(video_path: str, audio_path: str, gpu_id: str,
                     sharding_count: int = 8, gfpgan_upscale: int = 2, gfpgan_version: str = "1.3") -> str:

    """
    :param video_path: 音频文件
    :param audio_path: 视频文件
    :param gpu_id: gpu编号
    :param sharding_count: 源视频分片的数(默认值：8)
    :param gfpgan_upscale: 高分参数(默认值：2)
    :param gfpgan_version: 高分参数(默认值：1.3)
    :return:
    """
    work_path = '/home/ubuntu/winjean/GFPGAN-master'
    task_name = uuid.uuid4().hex
    video_name = video_path.split("/")[-1]
    audio_name = audio_path.split("/")[-1]
    current_path = os.getcwd()
    os.chdir(work_path)

    # # 创建task_dir
    task_dir = os.path.join(work_path, "tasks", task_name)
    os.makedirs(task_dir, exist_ok=True)

    # 复制输入文件至task_dir
    dst_file_video = os.path.join(task_dir, video_name)
    dst_file_audio = os.path.join(task_dir, audio_name)
    shutil.copy(video_path, dst_file_video)
    shutil.copy(audio_path, dst_file_audio)

    # 创建输出路径
    os.makedirs(os.path.join(task_dir, "output"), exist_ok=True)

    # 运行超分任务
    try:
        # GFPGAN
        video_path = gfpgan(task_name, video_name, audio_name, gpu_id, gfpgan_upscale, gfpgan_version, sharding_count)
    except Exception as e:
        print(e)
        logger.error(e)
        raise e

    os.chdir(current_path)
    return video_path


def handle_image_path(gpu_id: str, unprocessed_frame_path: str, processed_frame_path: str,
                      gfpgan_upscale: int, gfpgan_version: str):
    path_list = os.listdir(unprocessed_frame_path)
    path_list = sorted(path_list)

    # 创建多个线程处理图像
    threads = []
    for i, path in enumerate(path_list):
        input_image_path = os.path.join(unprocessed_frame_path, path)
        output_image_path = os.path.join(processed_frame_path, path)
        os.makedirs(output_image_path, exist_ok=True)

        input_image_abs_path = os.path.abspath(input_image_path)
        output_image_abs_path = os.path.abspath(output_image_path)

        t = threading.Thread(target=handle_image, args=(gpu_id, input_image_abs_path, output_image_abs_path,
                                                        gfpgan_upscale, gfpgan_version))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=300)


def handle_image(gpu_id: str, unprocessed_frames_folder_path: str, out_path: str,
                 gfpgan_upscale: int, gfpgan_version: str):
    command = [
        "/home/ubuntu/GFPGAN_Project/env/bin/python",
        "/home/ubuntu/winjean/GFPGAN-master/inference_gfpgan.py",
        "-i", unprocessed_frames_folder_path,
        "-o", out_path,
        "-v", gfpgan_version,
        "-s", str(gfpgan_upscale),
        "--only_center_face",
        "--bg_upsampler", "None"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    subprocess.run(command, capture_output=True, text=True, env=env)


def video_to_image(video_file: str, output_path: str, sharding_count: int):
    duration = get_video_duration(video_file)
    threads = []

    if duration == 0:
        raise Exception("视频时长为0，无法处理")

    if duration < 10:
        output_frame_path = os.path.join(output_path, "frame-01")
        os.makedirs(output_frame_path, exist_ok=True)
        video_to_image_with_fps(video_file, os.path.join(output_frame_path, "image_%04d.jpg"), 25)
    elif duration < 600:
        time_format = '00:{:02d}:{:02d}.00'
        for i in range(sharding_count):
            video_to_sub_image(duration, time_format, video_file, output_path, threads, i, sharding_count)

        for _thread in threads:
            _thread.join(timeout=300)

    else:
        raise Exception("视频时长超过600秒，无法处理")


def video_to_sub_image(duration: int, time_format: str, video_file: str, output_path: str, threads, n: int, m: int):
    time_1 = math.floor(duration*n/m)
    time_2 = math.floor(duration*(n+1)/m)
    time_start = time_format.format(time_1 // 60, time_1 % 60)
    time_end = time_format.format(time_2 // 60, time_2 % 60)
    output_frame_path = os.path.join(output_path, "frame-{i}".format(i=str(n).zfill(2)))
    os.makedirs(output_frame_path, exist_ok=True)
    t = threading.Thread(target=video_to_image_with_time_fps,
                         args=(video_file, os.path.join(output_frame_path, "image_%04d.jpg"), time_start, time_end, 25))
    threads.append(t)
    t.start()


# 获取视频时长
def get_video_duration(video_file: str) -> int:
    ffprobe_command = [
        'ffprobe',
        '-v', 'quiet',  # 不显示日志信息
        '-print_format', 'json',  # 输出格式为 JSON
        '-show_streams',  # 显示流信息
        video_file  # 输入视频文件
    ]

    # 调用 ffprobe 命令并捕获输出
    result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # 解析 JSON 输出
    info = json.loads(output)

    # 获取视频流的持续时间
    duration = None
    for stream in info['streams']:
        if stream['codec_type'] == 'video':
            duration = float(stream['duration'])
            break

    if duration is not None:
        import math
        print("视频时长:", duration)
        return math.ceil(duration)
    else:
        return 0


# 根据指定帧率，将视频转换为图片
def video_to_image_with_fps(video_file: str, output_jpg: str, fps: int):
    ffmpeg_command = ['ffmpeg',
                      '-i', video_file,
                      '-vf', 'fps=' + str(fps),
                      output_jpg]
    subprocess.call(ffmpeg_command)


# 根据指定帧率，将视频转换为图片
def video_to_image_with_time_fps(video_file: str, output_jpg: str, start_time: str, end_time: str, fps: int):
    ffmpeg_command = ['ffmpeg',
                      '-i', video_file,
                      '-ss', start_time, '-to', end_time,
                      '-vf', 'fps=' + str(fps),
                      output_jpg]
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"将视频 {video_file} 转换为图片时出错：{result.stderr}")


def merge_image_path_to_video(image_list_path: str, output_video: str):
    path_list = os.listdir(image_list_path)
    path_list = sorted(path_list)
    threads = []
    for i, path in enumerate(path_list):
        image_merge_path = os.path.join(image_list_path, path, "restored_imgs", "image_%04d.jpg")
        output_subsection_video = os.path.join(output_video, "video_{i}.mp4".format(i=str(i).zfill(2)))

        t = threading.Thread(target=merge_video, args=(image_merge_path, output_subsection_video, 25))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=300)


# 根据图片合成视频
def merge_video(input_images: str, output_video: str, fps: int):
    ffmpeg_command = ['ffmpeg',
                      '-i', input_images,
                      '-y',
                      '-r', str(fps),  # 帧率，可以根据需要调整
                      # '-c:v', 'libx264',  # 指定视频编解码器使用 libx264 编码
                      output_video]
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"将图片{input_images}合成视频时出错：{result.stderr}")


# 获取带拼接视频列表
def get_video_list_file(video_files, list_file_path):
    path_list = os.listdir(video_files)
    path_list = sorted(path_list)
    with open(list_file_path, 'w') as f:
        for i, video_file in enumerate(path_list):
            f.write(f"file '{video_file}'\n")


# 拼接视频
def concatenate_videos(video_files: str, output_file: str):
    # 创建视频列表文件
    list_file_path = os.path.join(video_files, 'video_list.txt')
    get_video_list_file(video_files, list_file_path)

    # 构建 ffmpeg 命令
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file_path,
        '-c', 'copy',
        output_file
    ]

    # 调用 ffmpeg 命令
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"拼接视频{list_file_path}时出错：{result.stderr}")


def merge_audio_video(input_audio_file: str, input_video_file: str, final_processed_output_video: str):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_video_file,
        "-i", input_audio_file,
        "-map", "0",
        "-map", "1",
        "-c:v", "copy",
        "-shortest",
        final_processed_output_video
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"将视频{input_video_file}和音频{input_audio_file}合并时出错：{result.stderr}")


if __name__ == '__main__':
    start_time = time.time()
    vp = "/home/ubuntu/winjean/GFPGAN-master/input_audio.mp4"
    ap = "/home/ubuntu/winjean/GFPGAN-master/input_audio.wav"
    result_path = super_resolution(vp, ap, "0")
    print(f"处理后视频输出路径：{result_path}")
    print("耗时：", time.time() - start_time)
