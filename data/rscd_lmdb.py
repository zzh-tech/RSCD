from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
import torch
import lmdb
import pickle
from os.path import join
from .utils import Crop, ToTensor, prepare
from .distortion_prior import distortion_map


class RSCDDataset(Dataset):
    def __init__(self, datapath='./lmdb_dataset/', dataset_type='train', frames=8, num_ff=2, num_fb=2, crop_size=256,
                 verbose=False):
        dataset_name = 'rscd'
        if dataset_type == 'train':
            self.datapath_inp = join(datapath, '{}_train'.format(dataset_name))
            self.datapath_gt = join(datapath, '{}_train_gt'.format(dataset_name))
            f = open(join(datapath, '{}_info_train.pkl'.format(dataset_name)), 'rb')
            self.seqs_info = pickle.load(f)
            f.close()
            self.transform = transforms.Compose([Crop(crop_size), ToTensor()])
        elif dataset_type == 'valid':
            self.datapath_inp = join(datapath, '{}_valid'.format(dataset_name))
            self.datapath_gt = join(datapath, '{}_valid_gt'.format(dataset_name))
            f = open(join(datapath, '{}_info_valid.pkl'.format(dataset_name)), 'rb')
            self.seqs_info = pickle.load(f)
            f.close()
            self.transform = transforms.Compose([Crop(crop_size), ToTensor()])
        self.verbose = verbose
        self.seq_num = self.seqs_info['num']
        self.seq_id_start = 0
        self.seq_id_end = self.seq_num - 1
        self.frames = frames
        self.crop_h, self.crop_w = crop_size
        self.W = 640
        self.H = 480
        self.down_ratio = 1
        self.C = 3
        self.num_ff = num_ff
        self.num_fb = num_fb
        self.env_inp = lmdb.open(self.datapath_inp, map_size=1099511627776)
        self.env_gt = lmdb.open(self.datapath_gt, map_size=1099511627776)
        self.txn_inp = self.env_inp.begin()
        self.txn_gt = self.env_gt.begin()
        self.encoding_map = distortion_map(self.H, self.W, ref_row=(self.H - 1) / 2).numpy()[..., np.newaxis]
        assert self.encoding_map.shape == (self.H, self.W, 1), self.encoding_map.shape

    def get_index(self):
        seq_idx = random.randint(self.seq_id_start, self.seq_id_end)
        frame_idx = random.randint(0, self.seqs_info[seq_idx]['length'] - self.frames)

        return seq_idx, frame_idx

    def get_img(self, seq_idx, frame_idx, sample):
        code = '%03d_%08d' % (seq_idx, frame_idx)
        code = code.encode()
        img_inp = self.txn_inp.get(code)
        img_inp = np.frombuffer(img_inp, dtype='uint8')
        img_inp = img_inp.reshape(self.H, self.W, self.C)
        img_inp = np.concatenate((img_inp, self.encoding_map.copy()), axis=2)
        img_gt = self.txn_gt.get(code)
        img_gt = np.frombuffer(img_gt, dtype='uint8')
        img_gt = img_gt.reshape(self.H, self.W, self.C)
        sample['image'] = img_inp
        sample['label'] = img_gt
        sample = self.transform(sample)
        if self.verbose:
            print('code', code, 's2s', sample['s2s'], 'top', sample['top'], 'left', sample['left'], 'flip_lr',
                  sample['flip_lr'], 'flip_ud', sample['flip_ud'], 'rotate',
                  sample['rotate'])

        return sample['image'], sample['label']

    def __getitem__(self, idx):
        top = random.randint(0, int(self.H * self.down_ratio) - self.crop_h)
        left = random.randint(0, int(self.W * self.down_ratio) - self.crop_w)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        rotate_flag = random.randint(0, 1)
        s2s_flag = random.randint(0, 9)
        #         reverse_flag = random.randint(0, 1)
        sample = {'s2s': s2s_flag, 'top': top, 'left': left, 'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag,
                  'rotate': rotate_flag}
        seq_idx, frame_idx = self.get_index()
        imgs_inp = []
        imgs_gt = []
        for i in range(self.frames):
            img_inp, img_gt = self.get_img(seq_idx, frame_idx + i, sample)
            imgs_inp.append(img_inp)
            imgs_gt.append(img_gt)
        imgs_inp = torch.cat(imgs_inp, dim=0)
        imgs_gt = torch.cat(imgs_gt[self.num_fb:self.frames - self.num_ff], dim=0)
        imgs_inp[:, :3, :, :] = prepare(imgs_inp[:, :3, :, :], normalize=True)
        imgs_gt = prepare(imgs_gt, normalize=True)
        return imgs_inp, imgs_gt

    def __len__(self):
        return self.seqs_info['length'] - (self.frames - 1) * self.seqs_info['num']


class Dataloader:
    def __init__(self, para, ds_type='train'):
        self.para = para
        path = join(para.data_root, para.dataset)
        self.dataset = RSCDDataset(path, ds_type, para.frames, para.future_frames, para.past_frames, para.patch_size)
        bs = self.para.batch_size
        ds_len = len(self.dataset)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=para.batch_size,
            shuffle=False,
            num_workers=para.threads,
            pin_memory=True,
            drop_last=True
        )
        self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len

    def reset(self):
        pass
