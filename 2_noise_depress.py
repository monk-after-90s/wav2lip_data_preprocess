# 音频降噪
import argparse
import os
import sys

sys.path.append(os.path.abspath("./DTLN"))
from typing import List
from DTLN.DTLN_model import DTLN_model
from DTLN.run_evaluation import process_file
import concurrent.futures
import traceback


def depress_nosie(audio_paths: List[str]):
    if not audio_paths: return
    # 降噪模型 todo 更换试试
    model = "DTLN/pretrained_model/model.h5"
    # determine type of model
    if model.find('_norm_') != -1:
        norm_stft = True
    else:
        norm_stft = False
    # create class instance
    modelClass = DTLN_model()
    # build the model in default configuration
    modelClass.build_DTLN_model(norm_stft=norm_stft)
    # load weights of the .h5 file
    modelClass.model.load_weights(model)

    for file in audio_paths:
        # look for .wav files
        if not file.endswith(".wav"):
            continue
        noise_depressed_audio_path = file.replace(audios_dir, noise_depressed_audios_dir)
        os.makedirs(os.path.dirname(noise_depressed_audio_path), exist_ok=True)
        # process each file with the model
        try:
            process_file(modelClass.model, file, noise_depressed_audio_path)
        except:
            traceback.print_exc()
        print(file + ' processed successfully! >>' + noise_depressed_audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--origin_audios_dir", help="The directory which contains wav audios", type=str,
                        default="dataset/origin_audios")
    parser.add_argument("--n_processes", help="Workers number", type=int, default=os.cpu_count() + 2)
    args = parser.parse_args()

    audios_dir = os.path.abspath(args.origin_audios_dir)
    if not os.path.isdir(audios_dir):
        raise ValueError("please input the path of a directory")
    parent_dir = os.path.dirname(audios_dir)
    audios_dir_name = os.path.basename(audios_dir)
    noise_depressed_audios_dir = os.path.join(parent_dir, audios_dir_name + "_ND")

    # 搜集音频文件路径
    audio_paths = []
    for root, dirs, files in os.walk(audios_dir):
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
        executor.map(depress_nosie, sub_tasks)
    print(f"noise_depressed_audios_dir={noise_depressed_audios_dir}")
