#!/usr/bin/env python
import tarfile
import librosa
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import random
os.chdir('/Users/kieraguan/Documents/coursework/6820/project/Download_data/Data/')
length=[]
mfccs=[]
start=0
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
                        mfcc = librosa.feature.mfcc(y=y, sr=sr,n_fft=800, hop_length=400, n_mfcc=24)
                        mfccs.append(mfcc)

    curlen=len(mfccs)-start
    start=len(mfccs)
    length.append(curlen)
    #print(mfccs.shape)
# path=filename[0]+'/_caustic_-20170306-smy/wav/'
# for name in os.listdir(path):
#     y, sr = librosa.load(path+name, sr=None)
#     if len(y)>=64000:
#         mid=len(y)//2
#         y=y[mid-32000:mid+32000]
#         mfcc = librosa.feature.mfcc(y=y, sr=sr,n_fft=800, hop_length=400, n_mfcc=24)
#         print(mfcc.shape)
mfccs=np.array(mfccs)

ohot=[[[1,0,0,0,0]]*161]*length[0]+[[[0,1,0,0,0]]*161]*length[1]+[[[0,0,1,0,0]]*161]*length[2]+\
     [[[0,0,0,1,0]]*161]*length[3]+[[[0,0,0,0,1]]*161]*length[4]
A=np.array(ohot)

index=[i for i in range(mfccs.shape[0])]
random.shuffle(index)
data=mfccs[index]
label=A[index]
x_test=data[:500]
y_test=label[:500]
x_train=data[500:55000]
y_train=label[500:55000]
x_val=data[55000:]
y_val=label[55000:]
print(x_test.shape,x_train.shape,x_val.shape)

f1=h5py.File("tr.hdf5","w")
f1.create_dataset('mfcc',data=x_train)
f1.create_dataset('label',data=y_train)
f2=h5py.File("test.hdf5","w")
f2.create_dataset('mfcc',data=x_test)
f2.create_dataset('label',data=y_test)
f3=h5py.File("val.hdf5","w")
f3.create_dataset('mfcc',data=x_val)
f3.create_dataset('label',data=y_val)
# # #for i in filename:
# plt.figure()
# plt.plot(mfccs)
# plt.show()