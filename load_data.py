#!/usr/bin/env python
import urllib.request
#import urllib2

import os
import re
os.chdir('/Users/kieraguan/Documents/coursework/6820/project/Download_data/French')#改变当前路径
#refiles=open('speech_files_path.txt','w+')#存储所有下载连接
#english='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
mainpath='http://www.repository.voxforge1.org/downloads/fr/Trunk/Audio/Main/16kHz_16bit/'
#german='http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/'
#french='http://www.repository.voxforge1.org/downloads/fr/Trunk/Audio/Main/16kHz_16bit/'
#italian='http://www.repository.voxforge1.org/downloads/it/Trunk/Audio/Main/16kHz_16bit/'
# russian='http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/'
#dutch='http://www.repository.voxforge1.org/downloads/Dutch/Trunk/Audio/Main/16kHz_16bit/'
def gettgz(url):
    response=urllib.request.urlopen(url)
    page_in=response.read().decode('utf-8')
    reg=r'href=.*.tgz'
    tgzre=re.compile(reg).findall(page_in)
      #找到所有.tgz文件
    for r in range(0,1000):
        i=tgzre[r]

        filename = i.split('>')
        filename = filename[1]
        filename = filename.replace('', '')
        print('Downloading: ' +str(r)+'file: '+ filename)  # 提示正在下载的文件
        downfile = mainpath + filename

        downfile = downfile.replace('', '')  # 得到每个文件的完整连接

        req = urllib.request.Request(downfile)  # 下载文件
        ur = urllib.request.urlopen(req).read()
        open(filename, 'wb').write(ur)


html=gettgz(mainpath)

#refiles.close()