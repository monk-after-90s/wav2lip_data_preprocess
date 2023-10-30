# 使用syncnet_python筛选
import argparse
import os
import sys

sys.path.append(os.path.abspath("./syncnet_python"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--frames_audios_dir",
                        help="The directory whose file tree contains frame png images and audios",
                        type=str,
                        default="dataset/frames_audios")
    parser.add_argument("--n_processes", help="Workers number", type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    frames_audios_dir = os.path.abspath(args.frames_audios_dir)
    if not os.path.isdir(frames_audios_dir):
        raise ValueError("please input the path of a directory")

    # 搜集视频文件路径
    video_paths = []
    for root, dirs, files in os.walk(frames_audios_dir):
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
    print(f"result: {frames_audios_dir}")
