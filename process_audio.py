import os
import scipy
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import cv2
import subprocess
import random
import sys
from utils import audio2stft, align_image, split_audio, read_audio, stft2audio

plt.switch_backend('agg')

SEED = 1023

AUDIO_DIR = "../dataset/dataset/trainset/audios/solo"
IMAGE_DIR = "../dataset/dataset/trainset/images/solo"
TRAIN_DIR = "/home/disk3/wcf/projects/std/train"
DEV_DIR = "/home/disk3/wcf/projects/std/dev"

LABELS = ["accordion", "acoustic_guitar", "cello", "flute", "saxophone", "trumpet", "violin", "xylophone"]
SIZE = [51, 48, 51, 43, 21, 38, 45, 44]

def R2P(R, Arg):
    return R * np.exp(1j*Arg)

def P2R(x):
    return np.abs(x), np.angle(x)

def save_audio(audio, filename):
    wavfile.write(filename, audio[0], audio[1])

def save_plot(spec, i):
    f,t,c = spec
    print("t: %d, f: %d" %(len(t), len(f)))
    plt.pcolormesh(t, f, np.abs(c))
    plt.title("STFT")
    plt.savefig('stft_ex-{0}.png'.format(i))

def prepare_test(root_path):
    #save_dir = '../dataset/dev'
    #if not os.path.isdir(save_dir):
    #os.mkdir(save_dir)

    #root_path = '../dataset/dataset/testset25'
    gt_audio_path = os.path.join(root_path, 'gt_audio')
    image_path = os.path.join(root_path, 'testimage')
    audios = []
    images = []
    count = 0
    gts = os.listdir(gt_audio_path)
    gt_mix = []
    for gt in gts:
        if gt.split('_')[-1].split('.')[0][:2] != 'gt':
        #find mix
            gt_mix.append(gt)

    img_dir_list = [os.path.join(image_path, x) for x in os.listdir(image_path)]
    for i in range(len(gt_mix)):
        print('read {}'.format(gt_mix[i]))
        #print(audios[1].shape)
        audio = read_audio(os.path.join(gt_audio_path, gt_mix[i]))
        print(audio[1].shape)
        audios = split_audio(os.path.join(gt_audio_path, gt_mix[i]))
        #save_audio([44100, np.concatenate(audios, axis=0).astype('int16')], 'cat.wav')
        #exit()
        #mstft = audio2stft(os.path.join(gt_audio_path, gt_mix[i]))
        mstfts = [audio2stft([44100, x], fix=True)[0] for x in audios]

        #save_audio([44100, np.concatenate([stft2audio(mstft, 1032*256)[1] for mstft in mstfts], axis=0).astype('int16')], 'cat_rec.wav')
        #exit()
        print('number of stfts: {}'.format(len(mstfts)))

        mstfts = np.stack(mstfts, axis=0)
        #print mstfts.shape
        #exit()
        #lstfts = mstfts#audio2stft(os.path.join(gt_audio_path, gt_mix[i].split('.')[0]+'_gt1.wav'))
        #rstfts = mstfts#audio2stft(os.path.join(gt_audio_path, gt_mix[i].split('.')[0]+'_gt2.wav'))

        img_path2 = os.path.join(image_path, gt_mix[i].split('.')[0])
        print(img_path2)
        img_names = os.listdir(img_path2)
        limg_names = [x for x in img_names if x[0] == 'l']
        rimg_names = [x for x in img_names if x[0] == 'r']
        rand_limg = random.sample(range(len(limg_names)), 1)
        rand_rimg = random.sample(range(len(rimg_names)), 1)
        rand_limg = [limg_names[x] for x in rand_limg]
        rand_rimg = [rimg_names[x] for x in rand_rimg]
        limg = [cv2.resize(cv2.imread(os.path.join(img_path2, _)), dsize=(224,224), interpolation=cv2.INTER_CUBIC) for _ in rand_limg]
        limg = np.stack(limg, axis=0)
        rimg = [cv2.resize(cv2.imread(os.path.join(img_path2, _)), dsize=(224,224), interpolation=cv2.INTER_CUBIC) for _ in rand_rimg]
        rimg = np.stack(rimg, axis=0)
        # output: T x H x W x 3
        return_data = {'limg':limg, 'rimg':rimg, 'mstft':[mstfts, audio, gt_mix[i].split('.')[0]]}
        yield return_data
        '''
        np.savez(
            os.path.join(save_dir, str(count)),
            #lstft = lstfts,
            #rstft = rstfts,
            limg = limg,
            rimg = rimg,
            mstft = mstfts)
        count += 1
        '''
def prepare_dev(save_dir, root_path):
    #save_dir = '../dataset/dev'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    #root_path = '../dataset/dataset/testset25'
    gt_audio_path = os.path.join(root_path, 'gt_audio')
    image_path = os.path.join(root_path, 'testimage')
    audios = []
    images = []
    count = 0
    gts = os.listdir(gt_audio_path)
    gt_mix = []
    for gt in gts:
        if gt.split('_')[-1].split('.')[0][:2] != 'gt':
            # find mix
            gt_mix.append(gt)

    img_dir_list = [os.path.join(image_path, x) for x in os.listdir(image_path)]
    for i in range(len(gt_mix)):
        print('read {}'.format(gt_mix[i]))
        audios = split_audio(os.path.join(gt_audio_path, gt_mix[i]))
        laudios = split_audio(os.path.join(gt_audio_path, gt_mix[i].split('.')[0]+'_gt1.wav'))
        raudios = split_audio(os.path.join(gt_audio_path, gt_mix[i].split('.')[0]+'_gt2.wav'))
        for j in range(len(audios)):
            mstft = audio2stft([44100, audios[j]])
            lstft = audio2stft([44100, laudios[j]])
            rstft = audio2stft([44100, raudios[j]])
            #mstft = audio2stft(os.path.join(gt_audio_path, gt_mix[i]))
            #lstft = audio2stft(os.path.join(gt_audio_path, gt_mix[i].split('.')[0]+'_gt1.wav'))
            #rstft = audio2stft(os.path.join(gt_audio_path, gt_mix[i].split('.')[0]+'_gt2.wav'))

            img_path2 = os.path.join(image_path, gt_mix[i].split('.')[0])
            print(img_path2)
            img_names = os.listdir(img_path2)
            limg_names = [x for x in img_names if x[0] == 'l']
            rimg_names = [x for x in img_names if x[0] == 'r']
            rand_limg = random.sample(range(len(limg_names)), 1)
            rand_rimg = random.sample(range(len(rimg_names)), 1)
            rand_limg = [limg_names[x] for x in rand_limg]
            rand_rimg = [rimg_names[x] for x in rand_rimg]
            limg = [cv2.resize(cv2.imread(os.path.join(img_path2, _)), dsize=(224,224), interpolation=cv2.INTER_CUBIC) for _ in rand_limg]
            limg = np.stack(limg, axis=0)
            rimg = [cv2.resize(cv2.imread(os.path.join(img_path2, _)), dsize=(224,224), interpolation=cv2.INTER_CUBIC) for _ in rand_rimg]
            rimg = np.stack(rimg, axis=0)
            # output: T x H x W x 3
            np.savez(
                os.path.join(save_dir, str(count)),
                lstft = lstft,
                rstft = rstft,
                limg = limg,
                rimg = rimg,
                mstft = mstft)
            count += 1

def prepare_single():
    save_path = '../dataset/train'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    audio_dirs = [os.path.join(AUDIO_DIR, label) for label in LABELS]
    img_dirs = [os.path.join(IMAGE_DIR, label) for label in LABELS]
    audios = []
    images = []
    count = 0
    for i in range(len(audio_dirs)):
        audios = [audio2stft(os.path.join(audio_dirs[i], str(k+1) + ".wav")) for k in range(SIZE[i])]
        img_path = [os.path.join(img_dirs[i], str(k+1)) for k in range(SIZE[i])]
        imgs = []
        print(img_path[0])
        for p in img_path:
            img_names = os.listdir(p)
            rand_img = random.sample(range(len(img_names)), 3)
            rand_img = [img_names[x] for x in rand_img]
            img = np.stack([cv2.imread(os.path.join(p, _)) for _ in rand_img], axis=0)
            # output: T x H x W x 3
            imgs.append(img)
        for j in range(len(audios)):
            np.savez(
                os.path.join(save_path, str(count)),
                image = imgs[j],
                stft = audios[j]
                )
            count += 1



def prepare_data(save_path, index_list):
    # set random seed
    audio_dirs = [os.path.join(AUDIO_DIR, label) for label in LABELS]
    img_dirs = [os.path.join(IMAGE_DIR, label) for label in LABELS]
    # random sample 4 example from each instrument
    audios = []
    images = []
    count = 0
    num = len(index_list[0][0])
    for i in range(len(audio_dirs)):
        for j in range(len(audio_dirs)):
            if i == j:
                break
            idx_i = index_list[0][i] #[np.random.randint(SIZE[i]-1)+1 for _ in range(4)]
            idx_j = index_list[1][j] #[np.random.randint(SIZE[j]-1)+1 for _ in range(4)]
            print(idx_i, idx_j)
            audio1 = [audio2stft(os.path.join(audio_dirs[i], str(_) + ".wav")) for _ in idx_i]
            audio2 = [audio2stft(os.path.join(audio_dirs[j], str(_) + ".wav")) for _ in idx_j]

            img_path1 = [os.path.join(img_dirs[i], str(_)) for _ in idx_i]
            img_path2 = [os.path.join(img_dirs[j], str(_)) for _ in idx_j]
            imgs1 = []
            for p in img_path1:
                imgs = os.listdir(p)
                rand_img = random.sample(range(len(imgs)), 3)
                rand_img = [imgs[x] for x in rand_img]
                img = np.stack([cv2.imread(os.path.join(p, _)) for _ in rand_img], axis=0)
                # output: T x H x W x 3
                imgs1.append(img)

            imgs2 = []
            for p in img_path2:
                imgs = os.listdir(p)
                rand_img = random.sample(range(len(imgs)), 3)
                rand_img = [imgs[x] for x in rand_img]
                img = np.stack([cv2.imread(os.path.join(p, _)) for _ in rand_img], axis=0)
                # output: T x H x W x 3
                imgs2.append(img)

            k_total = len(index_list[0][i])
            l_total = len(index_list[0][j])
            for k in range(k_total):
                for l in range(l_total):
                    im1, im2 = align_image(imgs1[k], imgs2[l])
                    # use f = np.load()
                    # check fields with f.files
                    np.savez(
                        os.path.join(save_path, str(count)),
                        left_image=im1,
                        left_stft=audio1[k],
                        right_image=im2,
                        right_stft=audio2[l])
                    count += 1


def main():
    try:
        root_path = sys.argv[1]
    except:
        raise RuntimeError('need an argument to specify the path to the data')
    prepare_dev('../data/std/dev-all', root_path)
    #print('save processed data in {}'.format(save_dir))
    #prepare_single()
    #prepare_dev()
    '''
    if not os.path.isdir(TRAIN_DIR):
        subprocess.call(["mkdir", TRAIN_DIR])
    if not os.path.isdir(DEV_DIR):
        subprocess.call(["mkdir", DEV_DIR])
    np.random.seed(SEED)
    #train_each = 7
    #dev_each = 3
    #total = (train_each + dev_each) * 2
    _list = [list(range(1,n+1)) for n in SIZE]
    #_list = [random.sample(range(1,n+1), total) for n in SIZE]
    print(_list)
    train_list = [[x[0:int(len(x)/2)] for x in _list], [x[int(len(x)/2):] for x in _list]]
    #print(train_list)
    #dev_list = [[x[2*train_each:2*train_each+dev_each] for x in _list], [x[2*train_each+dev_each:] for x in _list]]
    #print(dev_list)
    prepare_data(TRAIN_DIR, train_list)
    #prepare_data(DEV_DIR, dev_list)
    '''


if __name__ == '__main__':
    pass
    #main()
    #pass

