# 将视频转为帧图片保存
import sys
from typing import List

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")
from os import path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
import face_alignment
import torch


# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile: str, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    # 保存该视频的帧图片和声音的文件夹
    output_root = os.path.join(os.path.dirname(videos_dir), "frames_audios")
    output_dir = vfile.replace(videos_dir, output_root)[:-4]
    os.makedirs(output_dir, exist_ok=True)

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    i = -1
    for fb in batches:
        preds = fa[gpu_id].face_detector.detect_from_batch(torch.Tensor(np.asarray(fb).transpose(0, 3, 1, 2)))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = map(int, f[0][:-1])
            cv2.imwrite(path.join(output_dir, '{}.png'.format(i)), fb[j][y1:y2, x1:x2])


def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)


def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
    parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=16, type=int)
    parser.add_argument("--input_videos_dir",
                        help="Directory whose file tree contains mp4 files",
                        default="dataset/origin_noise_depressed_pieces",
                        type=str)
    # parser.add_argument("--output_dir", help="Directory which contains the preprocessed dataset", required=True)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.input_videos_dir)
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
    # 按GPU拆分任务
    fa: List[face_alignment.FaceAlignment] = [face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D,
                                                                           device='cuda:{}'.format(gpuid),
                                                                           face_detector='blazeface')
                                              for gpuid in range(args.ngpu)]
    template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
    print('Started processing for {} with {} GPUs'.format(videos_dir, args.ngpu))

    jobs = [(video_path, args, i % args.ngpu) for i, video_path in enumerate(video_paths)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    # print('Dumping audios...')#ToDo
    #
    # for vfile in tqdm(filelist):
    #     try:
    #         process_audio_file(vfile, args)
    #     except KeyboardInterrupt:
    #         exit(0)
    #     except:
    #         traceback.print_exc()
    #         continue
