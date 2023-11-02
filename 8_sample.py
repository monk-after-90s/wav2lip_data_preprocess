"""数据集抽样，得到数据集路径分别放进train.txt、val.txt、test.txt三个文件"""
import argparse
import os
import random

parser = argparse.ArgumentParser(description="数据集抽样，得到数据集路径分别放进train.txt、val.txt、test.txt三个文件")
parser.add_argument("--frames_audios_dir",
                    help="The directory whose file tree contains directories which contains frame jpg images and audio of videos",
                    type=str,
                    required=True)
parser.add_argument("--filelists_dir",
                    help="The directory which contains filelists output txt files",
                    type=str,
                    default="dataset/filelists")

args = parser.parse_args()
frames_audios_dir = os.path.abspath(args.frames_audios_dir)
if not os.path.isdir(frames_audios_dir):
    raise ValueError("please input the path of a directory")

# 搜集音画包文件夹
video_frams_audios_dirs = set()
for root, dirs, files in os.walk(frames_audios_dir):
    if "audio.wav" in files and any(".jpg" in file for file in files):
        video_frams_audios_dirs.add(os.path.relpath(root, args.frames_audios_dir))
if not video_frams_audios_dirs:
    raise FileNotFoundError("Empty directory")

video_frams_audios_dirs = list(video_frams_audios_dirs)
random.shuffle(video_frams_audios_dirs)
# 计算索引以分割列表
total_samples = len(video_frams_audios_dirs)
train_idx = int(total_samples * 0.8)
test_idx = train_idx + int(total_samples * 0.1)
# 分割列表
train_paths = video_frams_audios_dirs[:train_idx]
test_paths = video_frams_audios_dirs[train_idx:test_idx]
val_paths = video_frams_audios_dirs[test_idx:]


# 将数据写入文件
def write_to_file(file_path, paths):
    with open(file_path, 'w') as file:
        for path in paths:
            file.write(f'{path}\n')


# 指定文件路径
train_file = os.path.join(args.filelists_dir, 'train.txt')
test_file = os.path.join(args.filelists_dir, 'test.txt')
val_file = os.path.join(args.filelists_dir, 'val.txt')
# 写入文件
os.makedirs(args.filelists_dir, exist_ok=True)
write_to_file(train_file, train_paths)
write_to_file(test_file, test_paths)
write_to_file(val_file, val_paths)
print(f"result:{os.path.abspath(args.filelists_dir)}")
