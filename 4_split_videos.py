# run-to-split-each-video
import concurrent.futures
import argparse
import datetime
import os
import concurrent.futures
import subprocess
import uuid
from typing import List
import math
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_silence


def split_video(video_paths: List[str], split_mode=0, clip_duration=5):
    if split_mode == 1:
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
    elif split_mode == 0:
        from moviepy.video import VideoClip
        VideoClip.Clip._TEMP_FILES_PREFIX = str(uuid.uuid4())
        # Clip._TEMP_FILES_PREFIX = str(uuid.uuid4())
        for video_path in video_paths:
            video = VideoFileClip(video_path)
            audio = video.audio
            tmp_audio_file = f"/dev/shm/{uuid.uuid4()}{int(1000 * datetime.datetime.now().timestamp())}.wav"
            audio.write_audiofile(tmp_audio_file)
            try:
                audio_segment = AudioSegment.from_wav(tmp_audio_file)
                silence_intervals = detect_silence(audio_segment, min_silence_len=200, silence_thresh=-45)
            finally:
                os.remove(tmp_audio_file)

            # 保留静音部分，在静音中间设置切片点
            for i in range(len(silence_intervals)):
                sli_point = (silence_intervals[i][0] + silence_intervals[i][1]) / 2
                silence_intervals[i][0] = silence_intervals[i][1] = sli_point

            clips = []
            start_time = 0
            for silence_start, silence_end in silence_intervals:
                end_time = silence_start / 1000.0  # Convert milliseconds to seconds
                if start_time < end_time:
                    clip = video.subclip(start_time, end_time)
                    clips.append(clip)
                start_time = silence_end / 1000.0  # Convert milliseconds to seconds

            # Add the last segment of the video
            if start_time < video.duration:
                clips.append(video.subclip(start_time, video.duration))
            # 该文件的分割视频片段的保存文件夹位置
            split_videos_dir4one = video_path.replace(videos_dir, split_videos_dir, 1)[:-4]
            os.makedirs(split_videos_dir4one, exist_ok=True)
            for i, clip in enumerate(clips):
                clip.write_videofile(
                    os.path.join(split_videos_dir4one, f'{i}.mp4'), audio_codec="aac")

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将每个视频切成按声音停顿或者5s的多个片段")
    parser.add_argument("--source_videos_dir",
                        help="The directory whose file tree contains videos",
                        type=str,
                        default="dataset/origin_noise_depressed")
    parser.add_argument("--split_mode", help="切片模式：0：按声音停顿切片 1：:按固定5s间隔",
                        type=int,
                        default=0)
    parser.add_argument("--n_processes",
                        help="Workers number)",
                        type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.source_videos_dir)
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
        results = executor.map(split_video, sub_tasks, [args.split_mode] * len(sub_tasks))
        # 遍历结果
        for result in results:
            result
    print(f"result:{split_videos_dir}")
