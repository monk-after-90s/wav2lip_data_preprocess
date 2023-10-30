#!/usr/bin/python
# -*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import numpy
import time, subprocess, os, math, glob
import cv2
import python_speech_features
from scipy.io import wavfile
from SyncNetModel import *


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):  # vshift = 15

    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))  # (746, 1024) -> (776, 1024)

    dists = []
    # tmp1 = feat1[[0],:].repeat(win_size, 1)
    # tmp2 = feat2p[:win_size,:]
    # d = torch.nn.functional.pairwise_distance(tmp1, tmp2)
    # print(f"pairwise distance: {tmp1.shape}, {tmp2.shape} --> {len(d)}")
    for i in range(0, len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))

    return dists


# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout=0, num_layers_in_fc_layers=1024, verbose=False):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).cuda();
        self.verbose = verbose

    def separate_frames(self, opt, videofile):
        video_path = os.path.join(opt.video_dir, videofile)
        save_dir = os.path.join(opt.frame_dir, videofile[:-4])
        os.makedirs(save_dir, exist_ok=True)

        command = ("ffmpeg -loglevel error -y -i %s -threads 1 -f image2 %s" % (
            video_path, os.path.join(save_dir, '%06d.png')))
        output = subprocess.call(command, shell=True, stdout=None)

    def evaluate(self, opt, videofile):
        self.__S__.eval();
        # ========== ==========
        # Load video 
        # ========== ==========

        images = []

        flist = glob.glob(os.path.join(opt.frame_dir, videofile, '*.png'))  # frames
        flist.sort()

        for fname in flist:
            # images.append(cv2.imread(fname)) 
            images.append(self.pad_image_to_square(fname))

        im = numpy.stack(images, axis=3)  # (224, 224, 3, 752), 752는 프레임 개수
        im = numpy.expand_dims(im, axis=0)  # (1, 224, 224, 3, 752)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))  # (1, 3, 752, 224, 224)

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(opt.audio_dir, videofile + '.wav'))  # sample rate: 44100
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])  # (13, 3003)

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)  # (1, 1, 13, 3003)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio)) / sample_rate) != (float(len(images)) / 25):
            if self.verbose:
                print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." % (
                    float(len(audio)) / sample_rate, float(len(images)) / 25))

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + opt.batch_size))]
            im_in = torch.cat(im_batch, 0)  # (20, 3, 5, 224, 224)
            im_out = self.__S__.forward_lip(im_in.cuda());  # (20, 1024)
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in
                        range(i, min(lastframe, i + opt.batch_size))]
            cc_in = torch.cat(cc_batch, 0)  # (20, 1, 13, 20)
            cc_out = self.__S__.forward_aud(cc_in.cuda())  # (20, 1024)
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)  # (746, 1024)
        cc_feat = torch.cat(cc_feat, 0)  # (746, 1024)

        # ========== ==========
        # Compute offset
        # ========== ==========

        if self.verbose:
            print('Compute time %.3f sec.' % (time.time() - tS))

        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)  # after stack: (31, 746)
        mdist = torch.mean(torch.stack(dists, 1), 1)  # (31)

        minval, minidx = torch.min(mdist, 0)

        offset = (opt.vshift - minidx).item()
        conf = round((torch.median(mdist) - minval).item(), 4)
        dist = round(minval.item(), 4)
        print(offset, conf, dist)

        if self.verbose:
            print(f'AV offset: \t{offset}\nMin dist: \t{dist}\nConfidence: \t{conf}')

        return offset, conf, dist

    def evaluate_topK(self, opt, videofile, K=5):
        self.__S__.eval();

        # ========== ==========
        # Load video 
        # ========== ==========

        images = []

        flist = glob.glob(os.path.join(opt.frame_dir, videofile, '*.png'))  # frames
        flist.sort()

        for fname in flist:
            # images.append(cv2.imread(fname)) 
            images.append(self.pad_image_to_square(fname))

        im = numpy.stack(images, axis=3)  # (224, 224, 3, 752), 752는 프레임 개수
        im = numpy.expand_dims(im, axis=0)  # (1, 224, 224, 3, 752)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))  # (1, 3, 752, 224, 224)

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(opt.audio_dir, videofile + '.wav'))  # sample rate: 44100
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])  # (13, 3003)

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)  # (1, 1, 13, 3003)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio)) / sample_rate) != (float(len(images)) / 25):
            if self.verbose:
                print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." % (
                    float(len(audio)) / sample_rate, float(len(images)) / 25))

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + opt.batch_size))]
            im_in = torch.cat(im_batch, 0)  # (20, 3, 5, 224, 224)
            im_out = self.__S__.forward_lip(im_in.cuda());  # (20, 1024)
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in
                        range(i, min(lastframe, i + opt.batch_size))]
            cc_in = torch.cat(cc_batch, 0)  # (20, 1, 13, 20)
            cc_out = self.__S__.forward_aud(cc_in.cuda())  # (20, 1024)
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)  # (746, 1024)
        cc_feat = torch.cat(cc_feat, 0)  # (746, 1024)

        # ========== ==========
        # Compute offset
        # ========== ==========

        if self.verbose:
            print('Compute time %.3f sec.' % (time.time() - tS))

        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)  # after stack: (31, 746)
        mdist = torch.mean(torch.stack(dists, 1), 1)  # (31)

        # minval, minidx = torch.min(mdist,0)
        minvals, minidxs = torch.topk(-mdist, k=K, dim=0)
        minvals *= -1.

        offsets = [(opt.vshift - idx).item() for idx in minidxs]
        median_mdist = torch.median(mdist)
        confs = [round((median_mdist - val).item(), 4) for val in minvals]
        dists = [round(val.item(), 4) for val in minvals]

        if self.verbose:
            numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(f'AV offset: \t{offsets}\nMin dist: \t{dists}\nConfidence: \t{confs}')

        return offsets, confs, dists

    def pad_image_to_square(self, imagepath):
        """
        @ Ref: https://webnautes.tistory.com/1652
        """

        image = cv2.imread(imagepath)
        Hi, Wi, Ci = image.shape

        if Hi < Wi:
            Wf = 224
            Hf = int(Wf * (Hi / Wi))
        else:
            Hf = 224
            Wf = int(Hf * (Wi / Hi))

        if max(Hi, Wi) < 224:
            # upsampling
            image = cv2.resize(image, (Wf, Hf), cv2.INTER_CUBIC)
        else:
            # downsampling
            image = cv2.resize(image, (Wf, Hf), cv2.INTER_AREA)

        padH = 224 - Hf
        padW = 224 - Wf
        top, bottom = padH // 2, padH - (padH // 2)
        left, right = padW // 2, padW - (padW // 2)

        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # cv2.imwrite('./data/before.jpg', image)
        # cv2.imwrite('./data/after.jpg', padded_image)

        return padded_image

    def extract_feature(self, opt, videofile):

        self.__S__.eval();

        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images) - 4
        im_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + opt.batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========
        if self.verbose:
            print('Compute time %.3f sec.' % (time.time() - tS))

        return im_feat

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():
            self_state[name].copy_(param);
