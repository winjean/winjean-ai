import glob
import cv2
import numpy as np
import subprocess
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
    input_file_list = load_images(input_file)
    portrait_matting = pipeline(Tasks.portrait_matting, model='damo/cv_unet_image-matting')
    for i, input_file in enumerate(input_file_list):
        result = portrait_matting(input_file)
        cv2.imwrite(output_file.format(id=str(i + 1).zfill(4)), result[OutputKeys.OUTPUT_IMG])


def load_images(input_file: str):
    matched_files = glob.glob(input_file, recursive=True)
    return matched_files


def merge_images(matting_file: str, output_file: str):
    input_file_list = load_images(matting_file)

    matting_file = r"data\person\matting\image_{id}.png"
    background_file = r"data\person\background\image_{id}.jpg"

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


def main():
    person_video_file = r"data\person.mp4"
    background_video_file = r"data\background.mp4"

    person_fps, person_frame_count = get_frame(person_video_file)
    print(f"fps={person_fps}, frame_count={person_frame_count}")

    background_fps, background_frame_count = get_frame(background_video_file)
    print(f"fps={background_fps}, frame_count={background_frame_count}")

    min_frame_count = min(person_frame_count, background_frame_count)
    print(f"frame_count={min_frame_count}")

    person_output_jpg = r"data\person\person\image_%04d.jpg"
    video_image(person_video_file, person_output_jpg, min_frame_count)

    background_output_jpg = r"data\person\background\image_%04d.jpg"
    video_image(background_video_file, background_output_jpg, min_frame_count)

    input_file = r"data\person\person\image_*.jpg"
    output_file = r"data\person\matting\image_{id}.png"
    matting_image(input_file, output_file)

    matting_file = r"data\person\matting\image_*.png"
    output_merge_file = r"data\person\merge\image_{id}.jpg"
    merge_images(matting_file, output_merge_file)

    input_images = r"data\person\merge\image_%04d.jpg"
    output_video = r"data\person\merge.mp4"
    merge_video(input_images, output_video)


if __name__ == '__main__':
    main()
