import cv2
import os
from os.path import exists, dirname
import argparse
from model import Model
from para import Parameter
import torch
import torch.nn as nn
from data.utils import prepare, prepare_reverse
from data.distortion_prior import distortion_map
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help='the path of input video')
parser.add_argument('--dst', type=str, required=True, help='the path of output video')
parser.add_argument('--checkpoint', type=str, required=True, help='the path of JCD checkpoint')
args = parser.parse_args()


def inp_pre(img, h, w):
    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = prepare(img, normalize=True)
    img = np.concatenate((img, distortion_map(h, w, (h - 1) / 2.).numpy()[..., np.newaxis]), axis=2)
    img = img.transpose((2, 0, 1))[np.newaxis, :]
    return img


if not exists(args.src) and not exists(args.checkpoint):
    raise FileNotFoundError
if not dirname(args.dst) is '':
    os.makedirs(dirname(args.dst), exist_ok=True)

vidcap = cv2.VideoCapture(args.src)
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = 640
# height = 480
fps = int(vidcap.get(cv2.CAP_PROP_FPS)) // 3
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
size = (width, height)
video = cv2.VideoWriter(args.dst, fourcc, fps, size)
imgs = []
torch.cuda.empty_cache()
with torch.no_grad():
    model = Model(Parameter().args).model.cuda()
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for _ in range(frames):
        _, img = vidcap.read()
        imgs.append(inp_pre(img, height, width))
        if len(imgs) < 3:
            continue
        inp_imgs = np.concatenate(imgs)[np.newaxis, :]
        inp_imgs = torch.from_numpy(inp_imgs).float().cuda()
        out_imgs, _, _ = model(inp_imgs)
        del inp_imgs
        out_img = out_imgs[0]
        out_img = out_img.clamp(0, 1.0).squeeze(dim=0)
        out_img = out_img.detach().cpu().numpy().transpose((1, 2, 0))
        out_img = prepare_reverse(out_img, normalize=True).astype(np.uint8)
        video.write(out_img)
        imgs.pop(0)
vidcap.release()
video.release()
