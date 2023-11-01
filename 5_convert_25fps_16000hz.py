# 转视频帧率到25fps和音轨16000hz
import argparse
import concurrent.futures
import os
import subprocess
import json
import uuid
import shutil


def get_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",  # 选择第一个视频流
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    video_info = json.loads(result.stdout)
    video_frame_rate = eval(video_info["streams"][0]["r_frame_rate"])

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",  # 选择第一个音频流
        "-show_entries", "stream=sample_rate",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    audio_info = json.loads(result.stdout)
    audio_sample_rate = int(audio_info["streams"][0]["sample_rate"])

    return video_frame_rate, audio_sample_rate


def convert_video(video_path, output_path):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-r", "25",  # 设置帧率为 25fps
        "-ar", "16000",  # 设置音频采样率为 16000Hz
        "-ac", "1",  # 设置音频通道为单通道
        output_path,
        "-y"
    ]
    subprocess.run(cmd, check=True)


def transfer_videos_fps_sr(video_paths):
    """转换视频的帧率和音轨采样率"""
    tmp_file = os.path.join("/dev/shm", f"{str(uuid.uuid4())}.mp4")
    for video_path in video_paths:
        video_frame_rate, audio_sample_rate = get_video_info(video_path)
        if video_frame_rate != 25 or audio_sample_rate != 16000:
            convert_video(video_path, tmp_file)
            shutil.move(tmp_file, video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="转视频帧率到25fps和音轨16000hz")
    parser.add_argument("--videos_pieces_dir",
                        help="The directory which contains noise depressed videos pieces",
                        type=str,
                        default="dataset/origin_noise_depressed_pieces")
    parser.add_argument("--n_processes", help="Workers number", type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.videos_pieces_dir)
    if not os.path.isdir(videos_dir):
        raise ValueError("please input the path of a directory")

    # 搜集视频文件路径
    video_paths = []
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            name, extension = os.path.splitext(file)
            if extension != '.mp4':
                continue
            video_path = os.path.join(root, file)
            video_paths.append(video_path)
    if not video_paths:
        raise FileNotFoundError("Empty directory")

    # 分解成args.n_processes份子任务
    sub_tasks = [[] for _ in range(args.n_processes)]
    i = 0
    while video_paths:
        sub_tasks[i % args.n_processes].append(video_paths.pop())
        i += 1
    with concurrent.futures.ProcessPoolExecutor(args.n_processes) as executor:
        results = executor.map(transfer_videos_fps_sr, sub_tasks)
        # 遍历结果
        for result in results:
            result
    print(f"result: {videos_dir}")
