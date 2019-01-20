import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import argparse
from dataHelper import nussl, separation
import numpy as np
import cv2
import os
from tqdm import tqdm
import logging
import sys
#from tensorboardX import SummaryWriter
import subprocess
from layers import VisualAudioModel
from data import load_pro_data, VisualAudioDataset, SampleDataset, batchify, AverageMeter, SequenceDataset, load_data, load_test
from utils import stft2audio, audio2stft, save_audio, split_audio
logger = logging.getLogger()
import json
#from process_audio import prepare_test

#Writer = SummaryWriter('logs')
TRAIN_PATH = '/home/jihaozhe/data/std/train'
DEV_PATH = '/home/jihaozhe/data/std/dev-all'
MODEL_PATH = '/home/jihaozhe/data/std/models'


def add_args(parser):
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--test-batch-size', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--disp-iter',type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1023)
    parser.add_argument('--val-freq', type=int, default=1)
    parser.add_argument('--pretrain', type=str, default='models/load-350-best.mdl')#)#'20181231-dev-pretrain-load-350-10-epoch-16.mdl')#'load-350-epoch-10.mdl')#'load-100-0.78.mdl')#'20181230-adamax-binary-log-stft-fix-resnet-fix-order-load-100-epoch-35.mdl')#'20181230-adamax-binary-log-stft-fix-resnet-fix-order-load-100-epoch-43.mdl')#'adamax-epoch-400-no-val.mdl')#'adamax-epoch-400+300-no-val.mdl')#'20181226-adamax-binary-epoch-1000-load-1000-2500.mdl')#'adamax-epoch-400-no-val.mdl')#'sgd-sgd-epoch-1000-no-val.mdl')#'sgd-epoch-1000-no-val.mdl')#'sgd-2.mdl')
    parser.add_argument('--model-name', type=str, default='ratio-load-350-pretrain-dev.mdl')#'dev-pretrain-load-350-10')#'adamax-ratio-epoch-1000-load-5000-fix-resnet')#'adamax-binary-epoch-1000-load-1000-2500')#'adamax+sgd-epoch-400+300-no-val
    parser.add_argument('--validate', type=int, default=0)
    #parser.add_argument('--load-start', type=int, default=0)
    parser.add_argument('--run-test', type=int, default=1)
    parser.add_argument('--test-path', type=str, default='../data/std/testset25')
    parser.add_argument('--load-num',type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='adamax')

def set_defaults(args):
    if not os.path.isdir('result'):
        os.mkdir('result')
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    import time
    args.model_dir = os.path.join(MODEL_PATH, time.strftime("%Y%m%d-") + args.model_name)
    args.log_file = os.path.join('logs', args.model_name+'.txt')
    if args.pretrain != None:
        args.pretrain = os.path.join(MODEL_PATH, args.pretrain)
    #args.Writer = SummaryWriter(os.path.join('logs', args.model_name))

def train(args, model, loader, optimizer, epoch):
    total_loss = AverageMeter()
    model.train()
    for idx, ex in enumerate(loader):
        #with torch.cuda.device(0):
        audio = Variable(ex[0].cuda(async=True))
        rimg = Variable(ex[3].cuda(async=True))
        limg = Variable(ex[4].cuda(async=True))
        rgt = Variable(ex[1].cuda(async=True))
        lgt = Variable(ex[2].cuda(async=True))

        #print(audio.device)
        #print(rimg.device)
        #print(limg.device)
        raudio= model(audio, rimg)
        laudio = model(audio, limg)
        if idx == len(loader)-1:
            if False:#epoch % 30 == 0:
                logger.info('save training masks')
                cv2.imwrite('result/r_res-{}.png'.format(epoch), raudio[0].cpu().data.numpy()*255)
                cv2.imwrite('result/l_res-{}.png', laudio[0].cpu().data.numpy()*255)
                #cv2.imwrite('result/r_res_hard-{}.png'.format(epoch))
                cv2.imwrite('result/rmask.png'.format(epoch), rgt[0].cpu().data.numpy()*255)
                cv2.imwrite('result/lmask.png', lgt[0].cpu().data.numpy()*255)
        #cv2.imwrite('raudio2.png', (ex[5][1] / torch.max(ex[5][1]) * 255).numpy().astype('int'))
        #cv2.imwrite('raudio_rec.png', (ex[0][1] / torch.max(ex[0][1]) * 255 * rgt[1].cpu().data.float()).numpy().astype('int'))
        #cv2.imwrite('laudio_rec.png', (ex[0][1] / torch.max(ex[0][1]) * 255 * lgt[1].cpu().data.float()).numpy().astype('int'))
        #cv2.imwrite('laudio2.png', (ex[6][1] / torch.max(ex[6][1]) * 255).numpy().astype('int'))
        #exit(0)
        #print(raudio)

        #loss_right = - rgt * torch.log(raudio.clamp(min=1e-40)) - (1-rgt) * torch.log((1-raudio).clamp(min=1e-40))
        #loss_left = - lgt * torch.log(laudio.clamp(min=1e-40)) - (1-lgt) * torch.log((1-laudio).clamp(min=1e-40))
        loss_right = torch.log(torch.cosh(raudio - rgt))
        loss_left = torch.log(torch.cosh(laudio - lgt))
        batch_size = raudio.size(0)
        ex_size = raudio.size(1) * raudio.size(2)
        loss = loss_right + loss_left
        loss = loss.sum() / batch_size / ex_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step()

        total_loss.update(loss.item(), batch_size)

        #args.Writer.add_scalar('loss', total_loss.avg, (epoch*len(loader)+idx))
        if idx % args.disp_iter == 0:
            logger.info("train epoch: %d / %d | iter: %d / %d | loss = %.4f " %(epoch, args.epoch, idx, len(loader), total_loss.avg))


def Run_test(args, model, loader):
    model.eval()
    res_laudios = []
    res_raudios = []
    #gt_maudios = []
    names = []
    for idx, _ex in enumerate(loader):
        #if idx == 0:
        #    continue
        #print(len(ex))
        #print(len(ex[0]))
        #print(len(ex[0][0]))
        ex, ori_maudio, name = _ex[0]
        print(name)
        mstfts = [x[0] for x in ex]
        rimg = ex[0][1]
        limg = ex[0][2]
        laudio_list = []
        raudio_list = []
        # process ori audio
        ori_maudios = split_audio(ori_maudio)
        ori_mstfts = [audio2stft([44100, x], fix=True)[0] for x in ori_maudios]
        #print(limg[0])
        #cv2.imwrite('{}-left.png'.format(name), limg[0])#.numpy())
        #cv2.imwrite('{}-right.png'.format(name), rimg[0])#..numpy())
        #save_audio(ori_maudio, '{}.wav'.format(name))
        #exit()
        for i in range(len(mstfts)):
            # make batch 2
            N = 1032*256
            #print(limg.size())
            mstft_batch = torch.zeros(2, 256, 256).cuda()
            rimg_batch = torch.zeros(2,1,3,224,224).cuda()
            limg_batch = torch.zeros(2,1,3,224,224).cuda()

            mstft_batch[0] = mstfts[i]
            rimg_batch.data[0] = rimg
            limg_batch.data[0] = limg
            ratio_r = model(mstft_batch, rimg_batch)
            ratio_r = ratio_r[0].data.cpu()
            #print(ratio_r.size())
            #exit()
            #img_batch.data[0] = limg
            ratio_l = model(mstft_batch, limg_batch)
            ratio_l = ratio_l[0].data.cpu()

            lstft = ori_mstfts[i] * ratio_l.numpy()
            rstft = ori_mstfts[i] * ratio_r.numpy()
            #rstft = mstfts[i]#(mstfts[i] * ratio_r).numpy()
            #print(lstft.size())
            #exit()
            laudio = stft2audio(lstft, N)
            raudio = stft2audio(rstft, N)
            laudio_list.append(laudio[1])
            raudio_list.append(raudio[1])
        #print(len(laudio_list))
        #for d in laudio_list:
        #    print(d.shape)
        #exit()
        laudios = np.concatenate(laudio_list, axis=0)
        raudios = np.concatenate(raudio_list, axis=0)
        print(laudios.shape)
        print(ori_maudio[1].shape)
        res_laudios.append([44100, laudios[:ori_maudio[1].shape[0]]])
        res_raudios.append([44100, raudios[:ori_maudio[1].shape[0]]])
        #gt_maudios.append(ori_maudio)
        names.append(name)
        #save_audio([44100, laudios[:ori_maudio[1].shape[0]]], 'rec.wav')
        #save_audio(ori_maudio, 'ori.wav')
        print('successfully process one audio')
        #exit()
        # save audios
    return res_laudios, res_raudios, names

def calculate_sdr(r, l, gt_r, gt_l):
    gt = np.stack([gt_r, gt_l], axis=0)
    res = np.stack([r, l], axis=0)
    rvalue = separation.bss_eval_sources(gt, res, compute_permutation=True)
    return rvalue

def validate(args, model, loader, epoch, mode='dev', stft_mode='log', stop=25):
    Acc = AverageMeter()
    Sdr = AverageMeter()
    #Sir = AverageMeter()
    #Sar = AverageMeter()
    model.eval()
    count = 0
    for idx, ex in enumerate(tqdm(loader)):
        if count > stop:
            break
        #with torch.cuda.device(0):
        batch_size = ex[0].size(0)
        mix_stft = Variable(ex[0].cuda(async=True))
        rimg = Variable(ex[3].cuda(async=True))
        limg = Variable(ex[4].cuda(async=True))

        #print(mix_stft.device)
        ratio_r = model(mix_stft, rimg)
        ratio_l = model(mix_stft, limg)


        gt_ratio_r = ex[1]
        gt_ratio_l = ex[2]
        ratio_r = (ratio_r).cpu().data
        ratio_l = (ratio_l).cpu().data


        gt_raudios = ex[7]
        gt_laudios = ex[8]


        random_sample = np.random.randint(batch_size)
        for j in range(batch_size):
            count += 1
            if count > stop:
                break
            if True:#j == random_sample:
                union = min(gt_raudios[j][1].shape[0], gt_laudios[j][1].shape[0])
                gt_raudios[j][1] = gt_raudios[j][1][:union]
                gt_laudios[j][1] = gt_laudios[j][1][:union]
                mix_audio = [gt_raudios[j][0], (gt_raudios[j][1] + gt_laudios[j][1]).astype('int16')]
                mix_S = audio2stft(mix_audio)[0]
                #if stft_mode == 'log':
                #    rstft = np.exp(mix_S) * ratio_r[j].numpy()
                #    lstft = np.exp(mix_S) * ratio_l[j].numpy()
                #else:
                rstft = mix_S * ratio_r[j].numpy()
                lstft = mix_S * ratio_l[j].numpy()

                #audio = stft2audio(mix_S, union)
                raudio = stft2audio(rstft, union)
                laudio = stft2audio(lstft, union)
                if True:#args.validate:
                    cv2.imwrite('result/{}_r_res-{}.png'.format(mode, idx), ratio_r[j].cpu().data.numpy()*255)
                    cv2.imwrite('result/{}_rmask-{}.png'.format(mode, idx), gt_ratio_r[j].cpu().data.numpy()*255)
                    cv2.imwrite('result/{}_l_res-{}.png'.format(mode, idx), ratio_l[j].cpu().data.numpy()*255)
                    cv2.imwrite('result/{}_lmask-{}.png'.format(mode, idx), gt_ratio_l[j].cpu().data.numpy()*255)


                    #save_audio(raudio, 'result/{}_r_rec-{}.wav'.format(mode, idx))
                    #save_audio(gt_raudios[j], 'result/{}_r_gt-{}.wav'.format(mode, idx))
                    #save_audio(laudio, 'result/{}_l_rec-{}.wav'.format(mode, idx))
                    #save_audio(gt_laudios[j], 'result/{}_l_gt-{}.wav'.format(mode, idx))
                rvalue = calculate_sdr(raudio[1], laudio[1], gt_raudios[j][1], gt_laudios[j][1])
                print(rvalue)
                Sdr.update((rvalue[0][0] + rvalue[0][1])/2.0, 1)
    logger.info(" %s validate | epoch: %d | Average Sdr: %.4f" %(mode, epoch, Sdr.avg))
    return Sdr.avg


def init_optim(args, model):
    # TODO: fix resnet feature extract?
    if args.optimizer == 'sgd':
        logger.info('use optimizer: {}'.format(args.optimizer))
        visual_params = [p for p in model.module.resnet.parameters()]
        #visual_params = [p for p in model.module.right_resnet.parameters()] + [p for p in model.module.left_resnet.parameters()]

        audio_params = [p for p in model.module.unet_layer.parameters()] + [p for p in model.module.out.parameters()]
        #audio_params = [p for p in model.module.unet_layer.parameters()] + [p for p in model.module.out_left.parameters()] + [p for p in model.module.out_right.parameters()]
        optimizer = optim.SGD([{'params':visual_params, 'lr':1e-4},
                                {'params':audio_params, 'lr':1e-3}], momentum=0.9)
    if args.optimizer == 'adamax':
        logger.info('use optimizer: {}'.format(args.optimizer))

        params = [p for p in model.module.unet_layer.parameters()] + [p for p in model.module.out.parameters()] + [p for p in model.module.resnet.conv3x3.parameters()]
        #params = [p for p in model.module.unet_layer.parameters()] + [p for p in model.module.out_left.parameters()] + [p for p in model.module.out_right.parameters()]
        optimizer = optim.Adamax(params)
    #params = model.parameters()

    #optimizer = optim.Adam(params)
    return optimizer

def formatting_save(res_l, res_r, names, save_dir):
    res_dir = os.path.join(save_dir, 'result_audio')
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    result_json = {}
    for i, name in enumerate(names):
        save_audio(res_l[i], os.path.join(res_dir, name+'_seg1.wav'))
        save_audio(res_r[i], os.path.join(res_dir, name+'_seg2.wav'))
        data = []
        data.append({'position':0, 'audio':os.path.join('result_audio', name+'_seg1.wav')})
        data.append({'position':1, 'audio':os.path.join('result_audio', name+'_seg2.wav')})
        result_json[name+'.mp4'] = data
    with open(os.path.join(save_dir, 'result_json', 'result.json'), 'w') as f:
        json.dump(result_json, f)
        print('write to result.json')



def main(args):
    if args.run_test:
        test_exs = load_test(args, args.test_path)
        #exit()
        test_set = SequenceDataset(test_exs, test=True)
        test_sampler = RandomSampler(test_set)
        test_loader = DataLoader(
            test_set,
            batch_size = 1,
            num_workers = args.workers,
            collate_fn = batchify,
            pin_memory = True
            )
        #exit()
    else:
        train_exs = load_data(args, TRAIN_PATH, 'train')
        logger.info('load train examples: %d' %len(train_exs))
        dev_exs = load_data(args, DEV_PATH, 'dev')
        logger.info('load dev examples: %d' %len(dev_exs))
        train_set = SampleDataset(train_exs)
        train_sampler = RandomSampler(train_set)
        train_loader = DataLoader(
                train_set,
                batch_size = args.batch_size,
                num_workers = args.workers,
                collate_fn = batchify,
                pin_memory = True
                )
        dev_set = SequenceDataset(dev_exs)
        dev_sampler = RandomSampler(dev_set)
        dev_loader = DataLoader(
                dev_set,
                batch_size = args.test_batch_size,
                num_workers = args.workers,
                collate_fn = batchify,
                pin_memory = True
                )

    if args.pretrain == None:
        logger.info('Initialize from scatch...')
        # Run task on all available GPUs
        if torch.cuda.is_available():
            model = VisualAudioModel()

            if True:#torch.cuda.device_count() > 1:
                logger.info("Use "+str(torch.cuda.device_count())+" GPUs!")
                model = nn.DataParallel(model)

            #with torch.cuda.device(0):
            model = model.cuda()
            #model = model.module
            logger.info('Model Created on GPUs.')


    else:
        logger.info('Load from pretrained model {}...'.format(args.pretrain))
        if torch.cuda.is_available():
            model = torch.load(args.pretrain, map_location=lambda storage, loc: storage)

            if True:#torch.cuda.device_count() > 1:
                logger.info("Use "+str(torch.cuda.device_count())+" GPUs!")
                model = nn.DataParallel(model)

            model = model.cuda()
            #model = model.module
            logger.info('Model Created on GPUs.')



    optimizer = init_optim(args, model)

    logger.info('start training...')
    if args.validate or args.run_test:
        args.epoch = 1

    best_valid = -100
    for cur_epoch in range(args.epoch):
        if not args.validate and not args.run_test:
            train(args, model, train_loader, optimizer, cur_epoch)

        if cur_epoch %  args.val_freq == 0:
            if not args.validate and not args.run_test:
                torch.save(model.module, args.model_dir+'-epoch-{}'.format(cur_epoch)+'.mdl')
                logger.info('save epoch {}'.format(cur_epoch))
            if args.validate:
                validate(args, model, dev_loader, cur_epoch, 'dev')
                #sdr = validate(args, model, dev_loader, cur_epoch, 'dev')
                #best_valid = sdr
                #print('best valid: {}'.format(best_valid))
            if args.run_test:
                res_l, res_r, names = Run_test(args, model, test_loader)
                formatting_save(res_l, res_r, names, args.test_path)
if __name__ == '__main__':
    #subprocess.call('export', 'CUDA_VISIBLE_DEVICES=4,5')
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    #print(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    #set_defaults(args)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                                            '%m/%d/%Y %I:%M:%S %p')

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    args.log_file = os.path.join('logs', args.model_name+'.txt')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(args.log_file, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    main(args)

