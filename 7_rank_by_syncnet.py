# 使用syncnet_python进行打分分组
import argparse
import os
import sys

sys.path.append(os.path.abspath("./syncnet_python"))
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
from syncnet_python.newSyncNetInstance import SyncNetInstance
import math
import shutil
import torch


def gpu_work(video_frams_audios_dir: str, args, gpuid):
    """单个GPU的工作"""
    if not video_frams_audios_dir:
        return
    offset, conf, dist = sync_nets[gpuid].evaluate_frames_and_audio(
        video_frams_audios_dir,
        os.path.join(video_frams_audios_dir, "audio.wav"),
        args.batch_size,
        args.vshift
    )
    # 按照offset分类保存
    rank = min(math.floor(abs(offset)), 20)
    rank_dir = os.path.join(ranked_output_root, f"rank{rank}")
    output_dir = video_frams_audios_dir.replace(frames_audios_dir, rank_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(video_frams_audios_dir, output_dir, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用syncnet_python依据音画同步程度进行打分分组")
    parser.add_argument('--ngpu',
                        help='Number of GPUs across which to run in parallel',
                        default=torch.cuda.device_count(),
                        type=int)
    parser.add_argument("--frames_audios_dir",
                        help="The directory whose file tree contains frame png images and audios",
                        type=str,
                        default="dataset/frames_audios")
    parser.add_argument('--batch_size', type=int, default='1048576', help='Batch size when running syncnet_python')
    parser.add_argument('--vshift', type=int, default='15', help='syncnet_python argument')
    args = parser.parse_args()

    frames_audios_dir = os.path.abspath(args.frames_audios_dir)
    ranked_output_root = os.path.join(os.path.dirname(frames_audios_dir),
                                      "frames_audios_ranked_by_syncnet")
    if not os.path.isdir(frames_audios_dir):
        raise ValueError("please input the path of a directory")

    # 搜集音画包文件夹
    video_frams_audios_dirs = []
    for root, dirs, files in os.walk(frames_audios_dir):
        if "audio.wav" in files:
            video_frams_audios_dirs.append(root)
    if not video_frams_audios_dirs:
        raise FileNotFoundError("Empty directory")
    # 一个GPU一个sync_net模型
    sync_nets: List[SyncNetInstance] = []
    for gpuid in range(args.ngpu):
        s = SyncNetInstance(device='cuda:{}'.format(gpuid))
        s.loadParameters("syncnet_python/data/syncnet_v2.model")
        sync_nets.append(s)
    # 分解成args.ngpu份子任务
    jobs = [(video_frams_audios_dir, args, i % args.ngpu)
            for i, video_frams_audios_dir in enumerate(video_frams_audios_dirs)]
    gpus_pool_executor = ThreadPoolExecutor(args.ngpu)
    futures = [gpus_pool_executor.submit(gpu_work, *j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    print(f"result:{ranked_output_root}")
