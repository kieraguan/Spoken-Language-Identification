#!/usr/bin/env python
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import random
from PIL import Image


os.chdir('/Users/kieraguan/Documents/coursework/6820/project/Download_data/Data/')
length=[]
start=0
spectrogram=[]
# filename='German/4h-20100505-vgm/wav/fr-sb-758.wav'
# y, sr = librosa.load(filename, sr=None)
# print(y.shape)
# spec = librosa.stft(y, n_fft=256, hop_length=None, window='hann')
# mfcc = librosa.feature.mfcc(y=y, sr=sr,n_fft=256, hop_length=400, n_mfcc=24)
# mag_spec = np.abs(spec)
# step = (np.max(mag_spec) - np.min(mag_spec))
# new_index = np.uint8(255 * (mag_spec - np.min(mag_spec)) / step)
#
# plt.imshow(mag_spec**0.3, origin='lower',cmap='gray')
# to make it smoother, typically we use the power of 1/3 for visualization

# #plt.imshow(new_index, origin='lower')
# plt.show()
#
filename=['English','German','Dutch','Russian','Italian']
for file in filename:
    for recordings in os.listdir(file):
        if recordings!='.DS_Store':
            path=file+'/'+recordings+'/wav/'
            if os.path.isdir(path):
                for name in os.listdir(path):
                    print('MFCC:',path+name)
                    y, sr = librosa.load(path+name, sr=None)
                    if len(y)>=64000:
                        mid=len(y)//2

                        y=y[mid-32000:mid+32000]

                        spec = librosa.stft(y, n_fft=256, hop_length=None, window='hann')
                        mag_spec = np.abs(spec)  # magnitude spectrogram

                        step = (np.max(mag_spec) - np.min(mag_spec))
                        new_index = np.uint8(255 * (mag_spec - np.min(mag_spec)) / step)
                        spectrogram.append(new_index)
                        # mfcc = librosa.feature.mfcc(y=y, sr=sr,n_fft=800, hop_length=400, n_mfcc=24)
                        # mfccs.append(mfcc)
    curlen = len(spectrogram) - start
    start = len(spectrogram)
    length.append(curlen)
spectrogram=np.array(spectrogram)
ohot=[[1,0,0,0,0]]*length[0]+[[0,1,0,0,0]]*length[1]+[[0,0,1,0,0]]*length[2]+[[0,0,0,1,0]]*length[3]+[[0,0,0,0,1]]*length[4]
# ohot=[[1,0,0,0,0,0]]*length[0]+[[0,1,0,0,0,0]]*length[1]+[[0,0,1,0,0,0]]*length[2]+\
#      [[0,0,0,1,0,0]]*length[3]+[[0,0,0,0,1,0]]*length[4]+[[0,0,0,0,0,1]]*length[5]
A=np.array(ohot)
index=[i for i in range(spectrogram.shape[0])]
random.shuffle(index)
data=spectrogram[index]
label=A[index]
print(data.shape,label.shape)
# x_test=data[:1000]
# y_test=label[:1000]
# x_train=data[1000:56000]
# y_train=label[1000:56000]
# x_val=data[56000:]
# y_val=label[56000:]
x_test=data[:500]
y_test=label[:500]
x_train=data[500:55000]
y_train=label[500:55000]
x_val=data[55000:]
y_val=label[55000:]
print(x_test.shape,x_train.shape,x_val.shape)
f1=h5py.File("tr_data.hdf5","w")
f1.create_dataset('spec',data=x_train)
f1.create_dataset('label',data=y_train)
f2=h5py.File("test_data.hdf5","w")
f2.create_dataset('spec',data=x_test)
f2.create_dataset('label',data=y_test)
f3=h5py.File("val_data.hdf5","w")
f3.create_dataset('spec',data=x_val)
f3.create_dataset('label',data=y_val)
