import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from importlib import import_module
from .optimizer import Optimizer
from model import Model
from data import Data
import random
import numpy as np
from util.logger import Logger
from datetime import datetime
import pickle
import lmdb
import cv2
from os.path import join, dirname
from torch.nn.utils import clip_grad_norm_
from data.utils import prepare, prepare_reverse
from .metrics import psnr_calculate, ssim_calculate, lpips_calculate, AverageMeter
from .loss import loss_parse
from data.distortion_prior import distortion_map


class Trainer(object):
    def __init__(self, para):
        self.para = para

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para)
        logger.record_para()

        # training
        if not self.para.test_only:
            proc(self.para)

        # test
        test(self.para, logger)


# data parallel training
def proc(para):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # create logger
    logger = Logger(para)

    # create model
    logger('building {} model ...'.format(para.model), prefix='\n')
    model = Model(para).model
    model.cuda()
    logger('model structure:', model, verbose=False)

    # create criterion according to the loss function
    module = import_module('train.loss')
    criterion = getattr(module, 'Loss')(para).cuda()

    # create measurement according to metrics
    metrics_name = para.metrics
    module = import_module('train.metrics')
    metrics = getattr(module, metrics_name)().cuda()

    # create optimizer
    opt = Optimizer(para, model)

    # distributed data parallel
    model = nn.DataParallel(model)

    # create dataloader
    logger('loading {} dataloader ...'.format(para.dataset), prefix='\n')
    data = Data(para)
    train_loader = data.dataloader_train
    valid_loader = data.dataloader_valid

    # optionally resume from a checkpoint
    if para.resume:
        if os.path.isfile(para.resume_file):
            checkpoint = torch.load(para.resume_file, map_location=lambda storage, loc: storage.cuda(0))
            logger('loading checkpoint {} ...'.format(para.resume_file))
            logger.register_dict = checkpoint['register_dict']
            para.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            opt.optimizer.load_state_dict(checkpoint['optimizer'])
            opt.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            logger('no check point found at {}'.format(para.resume_file))

    # training and validation
    for epoch in range(para.start_epoch, para.end_epoch + 1):
        train(train_loader, model, criterion, metrics, opt, epoch, para, logger)
        valid(valid_loader, model, criterion, metrics, epoch, para, logger)

        # save checkpoint
        is_best = logger.is_best(epoch)
        checkpoint = {
            'epoch': epoch + 1,
            'model': para.model,
            'state_dict': model.state_dict(),
            'register_dict': logger.register_dict,
            'optimizer': opt.optimizer.state_dict(),
            'scheduler': opt.scheduler.state_dict()
        }
        logger.save(checkpoint, is_best)

        # reset DALI iterators
        train_loader.reset()
        valid_loader.reset()


def train(train_loader, model, criterion, metrics, opt, epoch, para, logger):
    model.train()
    logger('[Epoch {} / lr {:.2e}]'.format(
        epoch, opt.get_lr()
    ), prefix='\n')

    losses_meter = {}
    _, losses_name = loss_parse(para.loss)
    losses_name.append('all')
    for loss_name in losses_name:
        losses_meter[loss_name] = AverageMeter()

    measure_meter = AverageMeter()
    batchtime_meter = AverageMeter()
    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(train_loader), ncols=80)

    for inputs, labels in train_loader:
        # forward
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        if para.model.startswith('JCD'):
            imgs, masks, flows = outputs
            losses = criterion.rscd_forward(imgs, labels, masks, flows)
            outputs = imgs[0].unsqueeze(dim=1)
        else:
            losses = criterion(outputs, labels)
        measure = metrics(outputs.detach(), labels)
        for key in losses_name:
            losses_meter[key].update(losses[key].detach().item(), inputs.size(0))
        measure_meter.update(measure.detach().item(), inputs.size(0))

        # backward and optimize
        opt.zero_grad()
        loss = losses['all']
        loss.backward()
        # clip the grad
        clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        opt.step()

        # measure elapsed time
        batchtime_meter.update(time.time() - end)
        end = time.time()
        pbar.update(para.batch_size)

    pbar.close()
    # record info
    logger.register(para.loss + '_train', epoch, losses_meter['all'].avg)
    logger.register(para.metrics + '_train', epoch, measure_meter.avg)
    # show info
    logger('[train] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end - start, batchtime_meter.avg),
           timestamp=False)
    logger.report([[para.loss, 'min'], [para.metrics, 'max']], state='train', epoch=epoch)
    msg = '[train]'
    for key, meter in losses_meter.items():
        if key == 'all': continue
        msg += ' {} : {:4f};'.format(key, meter.avg)
    logger(msg, timestamp=False)

    # adjust learning rate
    opt.lr_schedule()


def valid(valid_loader, model, criterion, metrics, epoch, para, logger):
    model.eval()

    losses_meter = {}
    _, losses_name = loss_parse(para.loss)
    losses_name.append('all')
    for loss_name in losses_name:
        losses_meter[loss_name] = AverageMeter()

    measure_meter = AverageMeter()
    batchtime_meter = AverageMeter()
    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(valid_loader), ncols=80)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            if para.model.startswith('JCD'):
                imgs, masks, flows = outputs
                losses = criterion.rscd_forward(imgs, labels, masks, flows)
                outputs = imgs[0].unsqueeze(dim=1)
            else:
                losses = criterion(outputs, labels, valid_flag=True)
            measure = metrics(outputs.detach(), labels)
            for key in losses_name:
                losses_meter[key].update(losses[key].detach().item(), inputs.size(0))
            measure_meter.update(measure.detach().item(), inputs.size(0))

            # measure elapsed time
            batchtime_meter.update(time.time() - end)
            end = time.time()
            pbar.update(para.batch_size)

    pbar.close()
    # record info
    logger.register(para.loss + '_valid', epoch, losses_meter['all'].avg)
    logger.register(para.metrics + '_valid', epoch, measure_meter.avg)
    # show info
    logger('[valid] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end - start, batchtime_meter.avg),
           timestamp=False)
    logger.report([[para.loss, 'min'], [para.metrics, 'max']], state='valid', epoch=epoch)
    msg = '[valid]'
    for key, meter in losses_meter.items():
        if key == 'all': continue
        msg += ' {} : {:4f};'.format(key, meter.avg)
    logger(msg, timestamp=False)


def test(para, logger):
    logger('{} image results generating ...'.format(para.dataset), prefix='\n')
    if not para.test_only:
        para.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if para.test_save_dir == None:
        para.test_save_dir = logger.save_dir
    datasetType = para.dataset
    lmdb_type = 'test'
    if para.dataset.startswith('fastec_rs'):
        B, H, W, C = 1, 480, 640, 3
    elif para.dataset.startswith('rscd'):
        B, H, W, C = 1, 480, 640, 3
    else:
        raise NotImplementedError

    modelName = para.model.lower()
    model = Model(para).model.cuda()
    checkpointPath = para.test_checkpoint
    checkpoint = torch.load(checkpointPath, map_location=lambda storage, loc: storage.cuda())
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    data_test_path = join(para.data_root, datasetType, datasetType[:-4] + lmdb_type)
    data_test_gt_path = join(para.data_root, datasetType, datasetType[:-4] + lmdb_type + '_gt')
    env_inp = lmdb.open(data_test_path, map_size=int(3e10))
    env_gt = lmdb.open(data_test_gt_path, map_size=int(3e10))
    txn_inp = env_inp.begin()
    txn_gt = env_gt.begin()
    # load dataset info
    data_test_info_path = join(para.data_root, datasetType, datasetType[:-4] + 'info_{}.pkl'.format(lmdb_type))
    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)

    # PSNR, SSIM recorder
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS = AverageMeter()
    timer = AverageMeter()
    results_register = set()

    for seq_idx in range(seqs_info['num']):
        logger('seq {:03d} image results generating ...'.format(seq_idx))
        torch.cuda.empty_cache()
        save_dir = join(para.test_save_dir, datasetType + '_results_test', '{:03d}'.format(seq_idx))
        os.makedirs(save_dir, exist_ok=True)  # create the dir if not exist
        start = 0
        end = para.test_frames
        while (True):
            input_seq = []
            label_seq = []
            for frame_idx in range(start, end):
                code = '%03d_%08d' % (seq_idx, frame_idx)
                code = code.encode()
                img_inp = txn_inp.get(code)
                img_inp = np.frombuffer(img_inp, dtype='uint8')
                img_inp = prepare(img_inp.reshape(H, W, C), normalize=True)
                img_inp = np.concatenate((img_inp, distortion_map(H, W, (H - 1) / 2.).numpy()[..., np.newaxis]),
                                          axis=2)
                img_gt = txn_gt.get(code)
                img_gt = np.frombuffer(img_gt, dtype='uint8')
                img_gt = prepare(img_gt.reshape(H, W, C), normalize=True)
                input_seq.append(img_inp.transpose((2, 0, 1))[np.newaxis, :])
                label_seq.append(img_gt.transpose((2, 0, 1))[np.newaxis, :])
            input_seq = np.concatenate(input_seq)[np.newaxis, :]
            label_seq = np.concatenate(label_seq)[np.newaxis, :]
            model.eval()
            with torch.no_grad():
                input_seq = torch.from_numpy(input_seq).float().cuda()
                label_seq = torch.from_numpy(label_seq).float().cuda()
                if para.model.startswith('JCD'):
                    time_start = time.time()
                    output_seq = model(input_seq)
                    time_end = time.time() - time_start
                    timer.update(time_end)
                    imgs, masks, flows = output_seq
                    output_seq = imgs[0].unsqueeze(dim=1)
                    output_seq = output_seq.clamp(0, 1.0).squeeze(dim=0)
                else:
                    time_start = time.time()
                    output_seq = model(input_seq).clamp(0, 1.0).squeeze(dim=0)
                    timer.update(time.time() - time_start)

            for frame_idx in range(para.past_frames, end - start - para.future_frames):
                img_inp = input_seq.squeeze()[frame_idx].squeeze()
                img_inp = img_inp.detach().cpu().numpy().transpose((1, 2, 0))[:, :, :3]
                img_inp = prepare_reverse(img_inp, normalize=True).astype(np.uint8)
                img_inp_path = join(save_dir, '{:08d}_input.png'.format(frame_idx + start))
                img_gt = label_seq.squeeze()[frame_idx].squeeze()
                img_gt = img_gt.detach().cpu().numpy().transpose((1, 2, 0))
                img_gt = prepare_reverse(img_gt, normalize=True).astype(np.uint8)
                img_gt_path = join(save_dir, '{:08d}_gt.png'.format(frame_idx + start))
                img_out = output_seq[frame_idx - para.past_frames]
                img_out = img_out.detach().cpu().numpy().transpose((1, 2, 0))
                img_out = prepare_reverse(img_out, normalize=True).astype(np.uint8)
                img_out_path = join(save_dir, '{:08d}_{}.png'.format(frame_idx + start, modelName))
                cv2.imwrite(img_inp_path, img_inp)
                cv2.imwrite(img_gt_path, img_gt)
                cv2.imwrite(img_out_path, img_out)
                if img_out_path not in results_register:
                    results_register.add(img_out_path)
                    PSNR.update(psnr_calculate(img_out, img_gt))
                    SSIM.update(ssim_calculate(img_out, img_gt))
                    LPIPS.update(lpips_calculate(img_out, img_gt))
            if end == seqs_info[seq_idx]['length']:
                break
            else:
                start = end - para.future_frames - para.past_frames
                end = start + para.test_frames
                if end > seqs_info[seq_idx]['length']:
                    end = seqs_info[seq_idx]['length']
                    start = end - para.test_frames

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Test LPIPS : {}'.format(LPIPS.avg))
    logger('Average time : {}'.format(timer.avg))

    if para.video:
        logger('{} video results generating ...'.format(para.dataset), prefix='\n')
        marks = ['Input', modelName, 'GT']
        path = join(para.test_save_dir, datasetType + '_results_test')
        for i in range(seqs_info['num']):
            logger('seq {:03d} video result generating ...'.format(i))
            img2video(path, (3 * W, 1 * H), seq_num=i, frames=seqs_info[i]['length'], save_dir=path, marks=marks,
                      fp=para.past_frames, ff=para.future_frames)


# generate video
def img2video(path, size, seq_num, frames, save_dir, marks, fp, ff, fps=10):
    file_path = join(save_dir, '{:03d}.avi'.format(seq_num))
    os.makedirs(dirname(save_dir), exist_ok=True)  # create the dir if not exist
    path = join(path, '{:03d}'.format(seq_num))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for i in range(fp, frames - ff):
        imgs = []
        for j in range(len(marks)):
            img_path = join(path, '{:08d}_{}.png'.format(i, marks[j].lower()))
            img = cv2.imread(img_path)
            img = cv2.putText(img, marks[j], (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            imgs.append(img)
        frame = np.concatenate(imgs, axis=1)
        video.write(frame)
    video.release()
