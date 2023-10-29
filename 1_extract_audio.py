# 提取音频
import argparse
from moviepy.editor import *
import os
import concurrent.futures


def extract_audio(video_path: str):
    audio_path = video_path.replace(videos_dir, audios_dir)[:-3] + "wav"
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--videos_dir", help="The directory which contains mp4 videos", type=str,
                        default="dataset/origin")
    parser.add_argument("--n_processes", help="Workers number", type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.videos_dir)
    if not os.path.isdir(videos_dir):
        raise ValueError("please input the path of a directory")
    parent_dir = os.path.dirname(videos_dir)
    videos_dir_name = os.path.basename(videos_dir)
    audios_dir_name = f"{videos_dir_name}_audios"
    audios_dir = os.path.join(parent_dir, audios_dir_name)

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
    with concurrent.futures.ProcessPoolExecutor(args.n_processes) as executor:
        executor.map(extract_audio, video_paths)
    print(f"result:{audios_dir}")
