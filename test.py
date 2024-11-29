import subprocess
import os

# 视频文件路径
input_video_file = r"E:\code\winjean\winjean-ai\video\demo.mp4"

# 输出视频文件路径
output_video_file = os.path.join(os.getcwd(), "video", "output_trimmed.mp4")

# 定义截取时长
duration = 30  # 截取前30秒

# 构建 ffmpeg 命令
ffmpeg_command = [
    r'D:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg',
    '-y',  # 覆盖输出文件而不询问
    '-i', input_video_file,  # 输入视频文件
    '-t', str(duration),  # 截取时长
    output_video_file  # 输出视频文件
]

# 调用 ffmpeg 命令
subprocess.call(ffmpeg_command)

print(f"已生成截取后的视频: {output_video_file}")
