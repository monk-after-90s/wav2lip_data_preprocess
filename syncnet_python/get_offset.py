#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from tqdm import tqdm
from newSyncNetInstance import *

# ==================== PARSE ARGUMENT ====================
parser = argparse.ArgumentParser(description="SyncNet")
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
parser.add_argument('--batch_size', type=int, default='20', help='')
parser.add_argument('--vshift', type=int, default='15', help='')

parser.add_argument('--separate_frames', action='store_true')
parser.add_argument('--video_dir', type=str,
                    default='/home/leee/data/HDTF/video-25fps')
parser.add_argument('--frame_dir', type=str,
                    # default='/home/leee/data/HDTF/frame-s3fd')
                    default='/home/leee/data/HDTF/frame')
parser.add_argument('--audio_dir', type=str,
                    default='/home/leee/data/HDTF/audio-16k')
opt = parser.parse_args()

os.makedirs(opt.frame_dir, exist_ok=True)


def main(opt):
    # Load model and file lists
    s = SyncNetInstance()
    s.loadParameters(opt.initial_model)
    print("Model %s loaded." % opt.initial_model)

    # [Optional] Separate and save video frames independantly
    if opt.separate_frames:
        flist = [f for f in os.listdir(opt.video_dir) if f.endswith('.mp4')]

        for idx, fname in enumerate(tqdm(flist, desc='separating')):
            s.separate_frames(opt, videofile=fname)

    flist = os.listdir(opt.frame_dir)
    flist.sort()

    # Compute offset(and confidence) with various vshift values
    records = []
    print(f"vshift is {opt.vshift}.")

    for idx, fname in enumerate(tqdm(flist, desc='computing offset')):  # order matters!: enumerate(tqdm())
        offset, conf, dist = s.evaluate(opt, videofile=fname)
        records.append([offset, conf, dist])

    columns = ["offset", "conf", "dist"]

    df = pd.DataFrame(records, index=flist, columns=columns)
    os.makedirs("./output", exist_ok=True)
    df.to_csv("./output/offset.csv", sep=',', na_rep='NaN')
    print("./output/offset.csv is saved.")


if __name__ == "__main__":
    main(opt)
