import subprocess
import os

"""
默认情况下，FFmpeg 会根据输出文件的扩展名选择合适的编解码器和容器格式
ffmpeg -h 查看详细说明
    -i input: 指定输入文件。
    -f format: 强制使用某种输出格式。
       v4l2
    -an：不提取音频
    -vn：不包含视频流
    -y: 覆盖输出文件而不询问。
        ffmpeg -i input.mp4 -y output.mp4
    -c[:stream_specifier] codec: 选择编解码器。
       例如，-c:v libx264 表示使用 H.264 视频编解码器，-c:a aac 表示使用 AAC 音频编解码器。
    -c:v copy: 复制视频流，不进行重新编码。
    -c:a copy: 复制音频流，不进行重新编码。
        ffmpeg -i input.mp4 -c:a aac output.mp4 指定音频编解码器
    -b:v bitrate: 设置视频比特率，
       例如 -b:v 1000k 表示设置视频比特率为 1000 kbps。
    -b:a bitrate: 设置音频比特率，例如 -b:a 128k 表示设置音频比特率为 128 kbps。
        ffmpeg -i input.mp4 -b:a 128k output.mp4
    -r fps: 设置帧率，
       例如 -r 30 表示设置帧率为 30 fps。
    -g <interval>：设置关键帧间隔
        ffmpeg -i input.mp4 -g 120 output.mp4
    -q:v <quality>：设置视频质量（0-31，0为最高质量）
        ffmpeg -i input.mp4 -q:v 2 output.mp4
    -q:a <quality>：设置音频质量
        ffmpeg -i input.mp4 -q:a 4 output.mp4
    -s size: 设置视频分辨率，例如 -s 1280x720 表示设置分辨率为 1280x720。
    -aspect aspect: 设置宽高比，例如 -aspect 16:9。
    -t duration: 设置输出文件的持续时间，例如 -t 30 表示输出文件为 30 秒。
        ffmpeg -i input.mp4 -ss 00:01:00 -t 00:00:30 output.mp4
    -ss position: 开始时间偏移，例如 -ss 00:00:30.00 表示从第 30 秒开始处理。
        ffmpeg -i input.mp4 -ss 00:01:00 output.mp4
    -to position: 结束时间偏移，例如 -to 00:01:00.00 表示处理到第 1 分钟结束。
    -vf filter_graph: 应用视频滤镜，
       例如 -vf "scale=1280:720" 表示将视频缩放至 1280x720。
    -vf crop=<width>:<height>:<x>:<y>: 裁剪视频
        ffmpeg -i input.mp4 -vf "crop=640:480:0:0" output.mp4
    -vf "movie=watermark.png [watermark]; [in][watermark] overlay=W-w-10:H-h-10 [out]"：添加水印
        ffmpeg -i input.mp4 -vf "movie=watermark.png [watermark]; [in][watermark] overlay=W-w-10:H-h-10 [out]" output.mp4
        "subtitles=subtitles.srt"：添加字幕      
        ffmpeg -i input_video.mp4 -vf "subtitles=subtitles.srt:force_style='FontName=Arial,FontSize=24,PrimaryColour=&H00FF0000&'" -c:a copy output_video.mp4          
    -ab 128k：设置音频比特率为 128 kbps
    -ac <channels>: 设置音频通道数。  例如 -ac 2
        ffmpeg -i input.mp4 -ac 2 output.mp4
    -af filter_graph: 应用音频滤镜，例如 -af "volume=0.5" 表示将音量降低至原音量的一半。
        ffmpeg -i input.mp4 -af "volume=0.5" output.mp4
    -ar <sample_rate>: 设置音频采样率  例如 -ar 44100
        ffmpeg -i input.mp4 -ar 44100 output.mp4
    -loglevel <level>: 设置日志级别，例如 quiet, panic, fatal, error, warning, info, verbose, debug, trace
        ffmpeg -i input.mp4 -loglevel info output.mp4
    -preset <preset>: 设置编码速度与压缩比的预设，例如 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow。
    -vsync: 控制视频帧同步方式
        ffmpeg -i input.mp4 -vsync vfr output.mp4
        0 或 passthrough：不进行任何帧同步，直接输出源视频的帧。
        1 或 cfr：Constant Frame Rate，固定帧率。FFmpeg 会将输出视频的帧率调整为指定的帧率。
        2 或 vfr：Variable Frame Rate，可变帧率。FFmpeg 会根据源视频的实际帧率生成输出文件。并非所有容器格式都支持可变帧率,使用vfr可能会增加处理时间
        3 或 drop：丢弃重复帧，仅适用于 MPEG-2 和 MPEG-1 编码。
        -1 或 auto：自动选择最合适的帧同步方式，通常默认为 vfr
    -hwaccel <method>：启用硬件加速（auto, cuda, vaapi, qsv, dxva2
        ffmpeg -i input.mp4 -hwaccel cuda output.mp4
    -map <stream_specifier>：选择输入流进行多路复用
        ffmpeg -i input1.mp4 -i input2.mp4 -map 0:v -map 1:a output.mp4
    -metadata <key=value>：设置输出文件的元数据 
        ffmpeg -i input.mp4 -metadata title="My Video" output.mp4
    -f segment -segment_time <duration> -segment_list <list_file>：将输出文件分割成多个段
        ffmpeg -i input.mp4 -f segment -segment_time 60 -segment_list segments.txt output%03d.mp4
    -threads <count>：设置多线程处理的线程数
        ffmpeg -i input.mp4 -threads 4 output.mp4     
"""

# 图片文件夹路径
# image_folder = os.path.join(os.getcwd(), "images", "png")

# 输出视频文件路径
output_video = os.path.join(os.getcwd(), "video")

# 获取图片文件列表
# images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

# 按文件名排序（如果需要）
# images.sort()

# 生成图片文件路径列表
# image_paths = [os.path.join(image_folder, img) for img in images]

# 构建 ffmpeg 命令
# input_pattern = os.path.join(image_folder, 'image%03d.jpg')

demo_video_file = r"..\video\demo.mp4"

# 提取音频
# ffmpeg -i input.mp4 -q:a 0 output.mp3
output_audio_only_file = os.path.join(os.getcwd(), "audio", "output_audio_only.mp3")
ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-y', '-i', demo_video_file, '-q:a', '0',
                  output_audio_only_file]


# 提取视频
# ffmpeg -i input.mp4 -ss 00: 00:10 -t 00: 00:30 -c copy output.mp4
output_video_only_file = os.path.join(os.getcwd(), "video", "output_video_only.mp4")
# ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-y', '-i', demo_video_file,
#                   '-an', '-c:v', 'copy', output_video_only_file]

# 裁剪视频片段
# ffmpeg -i input.mp4 -ss 00: 00:10 -t 00: 00:30 -c copy output.mp4
video_cutter_file = os.path.join(os.getcwd(), "video", "output_cutter.mp4")
# ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-y', '-i', demo_video_file,
#                   '-ss', '00:00:05', '-t', '00:00:10', '-c', 'copy', video_cutter_file]

# 添加水印
# ffmpeg -i input.mp4 -i watermark.png -filter_complex "overlay=10:10" output.mp4
watermark_file = os.path.join(os.getcwd(), "images", "watermark.png")
video_watermark_file = os.path.join(os.getcwd(), "video", "output_watermark.mp4")
# ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-y',
#                   '-i', demo_video_file,
#                   '-i', watermark_file,
#                   '-filter_complex', 'overlay=10:10', video_watermark_file]

# 提取单张图片
# ffmpeg -i input.mp4 -ss 00:00:10 -vframes 1 output.jpg
output_jpg = r"../image/aa.jpg"
# ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-y', '-i', demo_video_file,
#                   '-ss', '00:00:05', '-vframes', '1', output_jpg]

# 提取多张图片
# ffmpeg -i input.mp4 -vf fps=1 output_%03d.jpg
# ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:30 -vf fps=1 output_%03d.jpg
output_jpg = os.path.join(os.getcwd(), "images", "jpg", "output_%03d.jpg")
# ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-i', demo_video_file,
#                   '-ss', '00:00:02', '-t', '00:00:30', '-vf', 'fps=1', output_jpg]

# 图片生成视频
# ffmpeg -i image_%03d.jpg output.mp4
# input_jpg = os.path.join(os.getcwd(), "images", "jpg", "output_%03d.jpg")
# output_video_file = os.path.join(output_video, "output.mp4")
# ffmpeg_command = [r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg', '-i', input_jpg,
#                   '-y',
#                   '-t', '10',
#                   '-r', '6',  # 帧率，可以根据需要调整
#                   '-c:v', 'libx264',  # 指定视频编解码器使用 libx264 编码
#                   output_video_file]

# 合成视频
video_input = output_video_only_file
audio_input = output_audio_only_file
output_file = os.path.join(output_video, "output_merge.mp4")
# 构建 ffmpeg 命令
# ffmpeg_command = [
#     'ffmpeg',
#     '-y',
#     '-ss', '00:00:20',  # 开始时间
#     '-t', '00:00:20',   # 持续时间
#     '-i', video_input,  # 输入视频文件
#     '-ss', '00:00:20',  # 开始时间
#     '-t', '00:00:20',   # 持续时间
#     '-i', audio_input,  # 输入音频文件
#     '-c:v', 'copy',     # 复制视频流，不重新编码
#     '-c:a', 'aac',      # 使用 AAC 编码音频
#     '-strict', 'experimental',  # 允许实验性的编码器选项
#     output_file         # 输出文件
# ]


# 调用 ffmpeg 命令
subprocess.call(ffmpeg_command)

print(f"已生成")
