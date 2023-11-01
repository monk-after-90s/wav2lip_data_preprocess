# run-to-split-each-video-into-5s-videos
import concurrent.futures
import argparse
import os
import concurrent.futures
import subprocess
from typing import List
import math


def split_video(video_paths: List[str], clip_duration=5):
    for video_path in video_paths:
        # 获取视频的总时长（秒）
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
               "default=noprint_wrappers=1:nokey=1",
               video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        total_duration = float(result.stdout)

        # 计算需要分割成多少个片段
        num_clips = math.ceil(total_duration / clip_duration)

        # 该文件的分割视频片段的保存文件夹位置
        split_videos_dir4one = video_path.replace(videos_dir, split_videos_dir)[:-4]
        os.makedirs(split_videos_dir4one, exist_ok=True)

        # 分割视频
        for i in range(num_clips):
            start_time = i * clip_duration
            cmd = ["ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(clip_duration), "-c", "copy",
                   os.path.join(split_videos_dir4one, f"{i}.mp4"), "-y"]
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将每个视频切成5s的多个片段")
    parser.add_argument("--noise_depressed_videos_dir",
                        help="The directory which contains noise depressed videos",
                        type=str,
                        default="dataset/origin_noise_depressed")
    parser.add_argument("--n_processes", help="Workers number", type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.noise_depressed_videos_dir)
    if not os.path.isdir(videos_dir):
        raise ValueError("please input the path of a directory")
    split_videos_dir = os.path.join(os.path.dirname(videos_dir),
                                    os.path.basename(videos_dir) + "_pieces")

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
        results = executor.map(split_video, sub_tasks)
        # 遍历结果
        for result in results:
            result
    print(f"result:{split_videos_dir}")
