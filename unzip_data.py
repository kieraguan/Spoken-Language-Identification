import tarfile

import os

os.chdir('/Users/kieraguan/Documents/coursework/6820/project/Download_data/')
filename=['English','French','German','Dutch','Russian','Italian']

for name in filename:
    pa=name
    for file in os.listdir(pa):
        path=pa+'/'+file
        print(path)
        a=tarfile.is_tarfile(path)
        if a:
            tar = tarfile.open(path, "r")
            f = tar.extractall('Data/'+name)
            tar.close