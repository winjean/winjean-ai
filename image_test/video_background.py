import glob
import os
import cv2
import numpy as np
import subprocess
import multiprocessing
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


def video_image(video_file: str, output_jpg: str, frame_count: int):
    ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg',
                      '-i', video_file,
                      # '-ss', '00:00:00', '-t', '00:00:20',
                      '-vf', f'select=between(n\\,0\\,{frame_count-1})',
                      '-vsync', 'vfr',
                      # 'fps=25',
                      output_jpg]
    subprocess.call(ffmpeg_command)


def get_frame(video_file: str) -> (float, int):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def matting_image(input_file: str, output_file: str):
    portrait_matting = pipeline(Tasks.portrait_matting, model='damo/cv_unet_image-matting')
    input_file_list = load_images(input_file)
    for i, input_file in enumerate(input_file_list):
        result = portrait_matting(input_file)
        cv2.imwrite(output_file.format(id=str(i + 1).zfill(4)), result[OutputKeys.OUTPUT_IMG])


def load_images(input_file: str):
    matched_files = glob.glob(input_file, recursive=True)
    return matched_files


def merge_images(matting_file: str, output_file: str):
    input_file_list = load_images(matting_file)

    matting_file = directory + r"\matting\image_{id}.png"
    background_file = directory + r"\background\image_{id}.jpg"

    for i, input_file in enumerate(input_file_list):
        # 读取带有Alpha通道的前景图像
        foreground = cv2.imread(matting_file.format(id=str(i + 1).zfill(4)), cv2.IMREAD_UNCHANGED)

        # 读取背景图像
        background = cv2.imread(background_file.format(id=str(i + 1).zfill(4)))

        # 调整背景图像大小
        if background.shape[:2] != foreground.shape[:2]:
            background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

        # 分离前景图像的RGB通道和Alpha通道
        fg_rgb = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255.0  # 将Alpha通道归一化到0-1之间

        # 计算背景图像的权重
        background_weight = 1.0 - alpha_channel

        # 将前景图像和背景图像混合
        result = ((fg_rgb * alpha_channel[:, :, np.newaxis] + background * background_weight[:, :, np.newaxis]).astype(np.uint8))

        # 保存结果图像
        cv2.imwrite(output_file.format(id=str(i + 1).zfill(4)), result)


def merge_video(input_images: str, output_video: str):
    ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-i',
                      input_images,
                      '-y',
                      # '-t', '10',
                      '-r', '25',  # 帧率，可以根据需要调整
                      '-c:v', 'libx264',  # 指定视频编解码器使用 libx264 编码
                      output_video]
    subprocess.call(ffmpeg_command)


def delete_files_in_directory(directory: str):
    files = glob.glob(os.path.join(directory, '*'))

    for file in files:
        os.remove(file)


def background_replace(foreground_video: str, background_video: str):
    processes = []

    # get fps and frame count
    person_fps, person_frame_count = get_frame(foreground_video)
    print(f"fps={person_fps}, frame_count={person_frame_count}")

    background_fps, background_frame_count = get_frame(background_video)
    print(f"fps={background_fps}, frame_count={background_frame_count}")

    min_frame_count = min(person_frame_count, background_frame_count)
    print(f"frame_count={min_frame_count}")

    # get foreground image from video
    foreground_output_jpg = directory + r"\foreground\image_%04d.jpg"
    video_image(foreground_video, foreground_output_jpg, min_frame_count)

    # get background images from video
    background_output_jpg = directory + r"\background\image_%04d.jpg"
    video_image(background_video, background_output_jpg, min_frame_count)

    # matting image
    foreground_input_file = directory + r"\foreground\image_*.jpg"
    matting_output_file = directory + r"\matting\image_{id}.png"
    matting_image(foreground_input_file, matting_output_file)

    # delete foreground image
    p1 = multiprocessing.Process(target=delete_files_in_directory, args=(directory + r"\foreground",))
    processes.append(p1)
    p1.start()

    # merge image
    matting_input_file = directory + r"\matting\image_*.png"
    merge_output_file = directory + r"\merge\image_{id}.jpg"
    merge_images(matting_input_file, merge_output_file)

    # delete background image, merge image
    p2 = multiprocessing.Process(target=delete_files_in_directory, args=(directory + r"\background",))
    processes.append(p2)
    p2.start()

    p3 = multiprocessing.Process(target=delete_files_in_directory, args=(directory + r"\matting",))
    processes.append(p3)
    p3.start()

    # images to video
    merge_input_images = directory + r"\merge\image_%04d.jpg"
    video_output = directory + r"\merge.mp4"
    merge_video(merge_input_images, video_output)

    # delete merge image
    p4 = multiprocessing.Process(target=delete_files_in_directory, args=(directory + r"\merge",))
    processes.append(p4)
    p4.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    directory = r"data\person"
    background_replace(r"data\person.mp4", r"data\background.mp4")
