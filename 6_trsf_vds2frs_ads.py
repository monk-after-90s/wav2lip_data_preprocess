# 将视频转为帧图片保存 提取视频的音轨保存为wav文件、mel频谱文件
import shutil
import sys
import uuid
from typing import List

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")
from os import path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback
from tqdm import tqdm
import face_alignment
import torch
import audio

sys.path.append(os.path.abspath("./DTLN"))

from DTLN.DTLN_model import DTLN_model
from DTLN.run_evaluation import process_file


# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile: str, args, gpu_id, modelClass):
    # 保存该视频的帧图片和声音的文件夹
    output_dir = vfile.replace(videos_dir, output_root)[:-4]
    # 检查是否已经存在
    if os.path.exists(path.join(output_dir, 'audio_mel.npy')):
        print(f"'{output_dir}'之前已经完成，本次跳过。")
        return

    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    face_rects = []
    for fb in batches:
        preds = fa[gpu_id].face_detector.detect_from_batch(torch.Tensor(np.asarray(fb).transpose(0, 3, 1, 2)))

        for j, f in enumerate(preds):
            if f is None:
                continue
            x1, y1, x2, y2 = map(int, f[0][:-1])
            # 分辨率不达标
            if abs(x2 - x1) < args.resolution_ratio or abs(y2 - y1) < args.resolution_ratio:
                print(f"视频'{vfile}'因分辨率不达标被舍弃")
                return

            face_rects.append(fb[j][y1:y2, x1:x2])

    os.makedirs(output_dir, exist_ok=True)
    # 存为图片
    for i, face_rect in enumerate(face_rects):
        cv2.imwrite(path.join(output_dir, '{}.jpg'.format(i)), face_rect)
    process_audio_file(modelClass, vfile, path.join(output_dir, 'audio.wav'))


def process_audio_file(modelClass, vfile, wav_path):
    command = template.format(vfile, wav_path)
    # subprocess.run(command, shell=True)
    os.system(command)

    noise_depressed_audio_path = f"/dev/shm/{uuid.uuid4()}.wav"
    try:
        process_file(modelClass.model, wav_path, noise_depressed_audio_path)
    except:
        traceback.print_exc()
    else:
        shutil.move(noise_depressed_audio_path, os.path.abspath(wav_path))
    finally:
        if os.path.exists(noise_depressed_audio_path):
            os.remove(noise_depressed_audio_path)

    # 存mel频谱
    wav = audio.load_wav(wav_path, 16000)
    mel = audio.melspectrogram(wav).T  # (T, 80)
    mel_path = os.path.join(os.path.dirname(wav_path), "audio_mel.npy")
    np.save(mel_path, mel)


def mp_handler(job):
    vfile, args, gpu_id, modelClass = job
    try:
        process_video_file(vfile, args, gpu_id, modelClass)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将视频转为帧图片保存；提取视频的音轨保存为wav文件")
    parser.add_argument('--ngpu',
                        help='Number of GPUs across which to run in parallel',
                        default=torch.cuda.device_count(),
                        type=int)
    parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=8, type=int)
    parser.add_argument('--resolution_ratio',
                        help='Resolution ratio requirements on both x and y direction of a face',
                        default=288, type=int)
    parser.add_argument("--input_videos_dir",
                        help="Directory whose file tree contains mp4 files",
                        default="dataset/origin_noise_depressed_pieces",
                        type=str)
    # parser.add_argument("--output_dir", help="Directory which contains the preprocessed dataset", required=True)
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.input_videos_dir)
    output_root = os.path.join(os.path.dirname(videos_dir), "frames_audios")
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
    # 按GPU拆分任务 todo blazeface裁剪大小跟sfd不一样
    fa: List[face_alignment.FaceAlignment] = [face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D,
                                                                           device='cuda:{}'.format(gpuid),
                                                                           face_detector='sfd')
                                              for gpuid in range(args.ngpu)]
    # 降噪模型
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

    template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
    print('Started processing for {} with {} GPUs'.format(videos_dir, args.ngpu))

    jobs = [(video_path, args, i % args.ngpu, modelClass) for i, video_path in enumerate(video_paths)]
    imgs_pool_executor = ThreadPoolExecutor(args.ngpu)
    futures = [imgs_pool_executor.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    print(f"result:{output_root}")
