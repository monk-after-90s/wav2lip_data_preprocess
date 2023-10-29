# 生成降噪视频
import argparse
import os
import concurrent.futures
from typing import List
from moviepy.editor import VideoFileClip, AudioFileClip


def generate_noise_depressed_video(audio_paths: List[str]):
    for audio_path in audio_paths:
        video_path = audio_path.replace(noise_depressed_audios_dir, videos_dir)[:-3] + "mp4"
        # 加载视频文件
        video_clip = VideoFileClip(video_path)
        # 加载音频文件
        audio_clip = AudioFileClip(audio_path)
        # 将视频文件中的音频替换为新的音频
        video_clip = video_clip.set_audio(audio_clip)
        noise_depressed_video_path = audio_path.replace(noise_depressed_audios_dir,
                                                        noise_depressed_videos_dir)[:-3] + "mp4"
        # 保存输出视频文件
        os.makedirs(os.path.dirname(noise_depressed_video_path), exist_ok=True)
        video_clip.write_videofile(noise_depressed_video_path, codec='libx264', audio_codec='aac')
        print(noise_depressed_video_path + " generated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--videos_dir", help="The directory which contains mp4 videos", type=str,
                        default="dataset/origin")
    parser.add_argument("--noise_depressed_audios_dir",
                        help="The directory which contains wav audios whose noises has been depressed ",
                        type=str,
                        default="dataset/origin_audios_ND")
    parser.add_argument("--n_processes", help="Workers number", type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.videos_dir)
    noise_depressed_audios_dir = os.path.abspath(args.noise_depressed_audios_dir)
    if not os.path.isdir(videos_dir):
        raise ValueError("please input the path of a directory")
    if not os.path.isdir(noise_depressed_audios_dir):
        raise ValueError("please input the path of a directory")
    noise_depressed_videos_dir = os.path.join(os.path.dirname(videos_dir),
                                              os.path.basename(videos_dir) + "_noise_depressed")
    # 搜集音频文件路径
    audio_paths = []
    for root, dirs, files in os.walk(noise_depressed_audios_dir):
        for file in files:
            name, extension = os.path.splitext(file)
            if extension != '.wav':
                continue
            audio_path = os.path.join(root, file)
            audio_paths.append(audio_path)
    if not audio_paths:
        raise FileNotFoundError("Empty directory")

    # 分解成args.n_processes份子任务
    sub_tasks = [[] for _ in range(args.n_processes)]
    i = 0
    while audio_paths:
        sub_tasks[i % args.n_processes].append(audio_paths.pop())
        i += 1
    with concurrent.futures.ProcessPoolExecutor(args.n_processes) as executor:
        executor.map(generate_noise_depressed_video, sub_tasks)
    print(f"result:{noise_depressed_videos_dir}")
