import os
import scipy
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import cv2
import subprocess
plt.switch_backend('agg')

SEED = 1023
AUDIO_DIR = "../dataset/dataset/trainset/audios/solo"
IMAGE_DIR = "../dataset/dataset/trainset/images/solo"
SAVE_DIR = "/home/jhz/train"

LABELS = ["accordion", "acoustic_guitar", "cello", "flute", "saxophone", "trumpet", "violin", "xylophone"]
SIZE = [51, 48, 51, 43, 21, 38, 45, 44]

# audio config
FREQ = 44100
DURATION = 6
W = 1024
HOP = 512


def read_audio(filename):
    fs, data = wavfile.read(filename)
    #print(fs)
    #print(data.shape[0]/fs)
    return [fs, data]

def save_audio(audio, filename):
    wavfile.write(filename, audio[0], audio[1])

def Stft(audio):
    return abs(librosa.core.stft(audio[1].astype('float'), n_fft = W, hop_length=HOP))

def Istft(stft):
    sound_rec = librosa.core.istft(stft, win_length=W, hop_length=HOP)
    return [FREQ, sound_rec.astype('int16')]

def rec_audio(stft, ori_len):
    rec_audio = Istft(stft)
    if len(rec_audio[1]) < ori_len:
        rec = np.zeros(ori_len)
        rec[:len(rec_audio[1])] = rec_audio
    else:
        rec = rec_audio[:ori_len]
    return rec

def stft(x, fs, framesz, hop):
    framesamp = framesz
    hopsamp = hop
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
    for i in range(0, len(x)-framesamp, hopsamp)])
    return X.transpose()

def istft(X, fs, Len, hop):
    X = X.transpose()
    x = scipy.zeros(Len)
    framesamp = X.shape[1]
    hopsamp = hop
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return [fs, x.astype('int16')]

def R2P(R, Arg):
    return R * np.exp(1j*Arg)

def P2R(x):
    return np.abs(x), np.angle(x)

def DownSample(input_signal,src_fs,tar_fs):
    dtype = input_signal.dtype
    audio_len = len(input_signal)
    audio_time_max = 1.0*(audio_len-1) / src_fs
    src_time = 1.0 * np.linspace(0,audio_len,audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0,np.int(audio_time_max*tar_fs),np.int(audio_time_max*tar_fs)) / tar_fs
    output_signal = np.interp(tar_time,src_time,input_signal).astype(dtype)

    return output_signal

def DownSampleSTFT(stft, rate):
    dtype = stft.dtype
    f = stft.shape[0]
    t = stft.shape[1]
    new_f = int(f/rate)
    res = np.zeros((new_f, t), dtype=np.complex_)
    for col in range(t):
        real_sp = UpSample(stft[:, col].real, f*10, new_f*10, new_f)
        imag_sp = UpSample(stft[:, col].imag, f*10, new_f*10, new_f)
        res[:, col] = real_sp + 1j * imag_sp
    return res.astype(dtype)

def UpSampleSTFT(stft, rate):
    dtype = stft.dtype
    f = stft.shape[0]
    t = stft.shape[1]
    new_f = int(f*rate)
    res = np.zeros((new_f, t), dtype=np.complex_)
    for col in range(t):
        real_sp = UpSample(stft[:, col].real, f*10, new_f*10, new_f)
        imag_sp = UpSample(stft[:, col].imag, f*10, new_f*10, new_f)
        res[:, col] = real_sp + 1j * imag_sp
    return res.astype(dtype)


def UpSample(input_signal, src_fs, tar_fs, tar_len):
    dtype = input_signal.dtype
    src_len = input_signal.shape[0]
    src_time = 1.0 * np.linspace(0, src_len, src_len) / src_fs
    tar_time = 1.0 * np.linspace(0, tar_len, tar_len) / tar_fs
    output = np.interp(tar_time, src_time, input_signal).astype(dtype)
    return output

def split_audio(input_, N=1032*256):
    if len(input_) != 2:
        audio = read_audio(input_)
    else:
        audio = input_
    num = int(audio[1].shape[0] / N)
    sep = []
    for i in range(num):
        sep.append(audio[1][i*N:(i+1)*N])
    res = np.zeros((N,))
    res[0:(audio[1].shape[0] - num * N)] = audio[1][num*N:]
    sep.append(res)
    return sep

def audio2stft(input_, N=1032 * 256, fs=11000, w=510, hop=256, fix=False):
    if len(input_) == 2:
        audio = input_
    else:
        audio = read_audio(input_)
    # take N points from middle
    ori_ex = []
    ori_ex.append(audio[0])
    ori_len = audio[1].shape[0]
    if not fix:
        ori_ex.append(audio[1][int((ori_len-N)/2):int((ori_len-N)/2)+N])
    else:
        ori_ex.append(audio[1])
    # sample to desired freq
    sp_ex = []
    sp_ex.append(fs)
    sp_ex.append(DownSample(ori_ex[1], audio[0], fs))
    # stft
    S = stft(sp_ex[1], sp_ex[0], w, hop)
    S = S[0:w/2+1]
    return S, ori_ex

def stft2audio(S, ori_len, fs=11000, rec_fs=44100, w=510, hop=256):
    sp_len = int(ori_len * fs / rec_fs)
    # replicate the half top STFT
    S = np.concatenate([S, np.conjugate(S)[1:-1][::-1]], axis=0)
    rec_ex = istft(S, fs, sp_len, hop)
    rec_ex[0] = rec_fs
    rec_ex[1] = UpSample(rec_ex[1], fs, rec_fs, ori_len)
    return rec_ex


def align_image(img1, img2):
    max_h = max(img1.shape[1], img2.shape[1])
    if max_h > img1.shape[1]:
        s = []
        w = int(img1.shape[2] * max_h / img1.shape[1])
        for d in range(img1.shape[0]):
            img = cv2.resize(img1[d], dsize=(w, max_h), interpolation=cv2.INTER_CUBIC)
            s.append(img)
        img = np.stack(s, axis=0)
        v = (img, img2)
    else:
        s = []
        w = int(img2.shape[2] * max_h / img2.shape[1])
        for d in range(img2.shape[0]):
            img = cv2.resize(img2[d], dsize=(w, max_h), interpolation=cv2.INTER_CUBIC)
            s.append(img)
        img = np.stack(s, axis=0)
        v = (img1, img)
    return v


def demo():
    N = 1032 * 128
    ex_dir = os.path.join(AUDIO_DIR, LABELS[1], "1.wav")
    audio_ex1 = read_audio(ex_dir)
    audio_ex1[1] = audio_ex1[1][3000000:3000000+N]
    ori_len = audio_ex1[1].shape[0]
    print(audio_ex1[1].shape)
    sp_audio = []
    sp_audio.append(11000)
    sp_audio.append(DownSample(audio_ex1[1], 44100, 11000))
    print(sp_audio[1].shape)

    S1 = stft(sp_audio[1], sp_audio[0], 256,128)
    sp_len = sp_audio[1].shape[0]
    print(S1.shape)

    S_u = S1

    rec_ex = istft(S_u, 11000, sp_len, 128)
    print(rec_ex[1].shape[0])
    rec_ex[0] = 44100
    rec_ex[1] = UpSample(rec_ex[1], 11000, 44100, ori_len)

    print(rec_ex[1].shape)
    save_audio(audio_ex1, '../eval/gt_audio/1_gt1.wav')
    save_audio(rec_ex, '../eval/result_audio/1_seg1.wav')

def main():
    ex_dir = os.path.join(AUDIO_DIR, LABELS[1], "1.wav")
    S, audio = audio2stft(ex_dir)
    print(S.shape)
    ori_len = audio[1].shape[0]
    rec_audio = stft2audio(S, ori_len)
    save_audio(audio, '../eval/gt_audio/1_gt1.wav')
    save_audio(rec_audio, '../eval/result_audio/1_seg1.wav')



