import os
import re
import hashlib
import shutil
from time import time

import cv2
import glob

# backupPath = '/home/yaoqiang/data/data_collection/live_detect/yanan_align_sort_unique/spoof'
backupPath = '/home/yaoqiang/Downloads/DCGAN-tensorflow/samplesEx_unique'
picDic = {}

regular = re.compile(r'^(.*)/.(jpg|jpeg|bmp|gif|png|JPG|JPEG|BMP|GIF|PNG)$')

def verification_suffix(img_name):
    if img_name.endswith('.jpg') | img_name.endswith('.png') |img_name.endswith('.bmp'):
        return True
    else:
        return False


def RemoverRePic(dirPath):

    quantity = 0

    for childPath in os.listdir(unicode(dirPath)):
        childPath = dirPath + '/' + childPath
        if os.path.isdir(childPath):
            quantity = + RemoverRePic(childPath)
        else:
            # if regular.match(childPath):
            if verification_suffix(childPath):
                pic = open(childPath, 'rb')
                picMd5 = hashlib.md5(pic.read()).hexdigest()
                pic.close()

                if picDic.has_key(picMd5):
                    # newPath = backupPath + '/' + hashlib.md5(childPath) \
                    # .hexdigest() + childPath[childPath.find('.'):]
                    # shutil.move(childPath, newPath)
                    # os.rename(childPath, newPath)
                    quantity = + 1
                else:
                    newPath = backupPath + '/' + hashlib.md5(childPath) \
                    .hexdigest() + childPath[childPath.find('.'):]
                    shutil.move(childPath, newPath)
                    picDic[picMd5] = childPath
    return quantity

# slove the problem of large image
def getSmallImagesSameName(dstpath):
    root_path = '/home/yaoqiang/data/tyres/20190117/dst_pics_full/'
    image_paths = glob.glob(root_path + "*.jpg")

    path_len = len(image_paths)
    print(path_len)

    # dstpath = '/home/yaoqiang/data/tyres/20190117/dst_pics_full_small/'

    for i in (range(path_len)):
        imagepath = image_paths[i]

        img_color = cv2.imread(imagepath)
        img_color = cv2.resize(img_color, (128,128))

        imgName = imagepath[imagepath.rfind('/')+1:]
        print(imgName)
        dstpic = os.path.join(dstpath,imgName)

        cv2.imwrite(dstpic, img_color)

    return


def RemoverRePicEx(dirPath):

    quantity = 0

    for childPath in os.listdir(unicode(dirPath)):

        root_path = '/home/yaoqiang/data/tyres/20190117/dst_pics_full'
        root_path = root_path + '/' + childPath

        childPath = dirPath + '/' + childPath
        if os.path.isdir(childPath):
            quantity = + RemoverRePic(childPath)
        else:

            # if regular.match(childPath):
            if verification_suffix(childPath):
                pic = open(childPath, 'rb')
                picMd5 = hashlib.md5(pic.read()).hexdigest()
                pic.close()

                if picDic.has_key(picMd5):
                    # newPath = backupPath + '/' + hashlib.md5(childPath) \
                    # .hexdigest() + childPath[childPath.find('.'):]
                    # shutil.move(childPath, newPath)
                    # os.rename(childPath, newPath)
                    quantity = + 1
                else:
                    newPath = backupPath + '/' + hashlib.md5(childPath) \
                    .hexdigest() + childPath[childPath.find('.'):]
                    # shutil.move(childPath, newPath)
                    shutil.move(root_path, newPath)
                    picDic[picMd5] = childPath
    return quantity


if __name__ == '__main__':

    # getSmallImagesSameName("/home/yaoqiang/data/tyres/20190117/dst_pics_full_small/")
    # RemoverRePicEx("/home/yaoqiang/data/tyres/20190117/dst_pics_full_small")
    # fname = "/home/yaoqiang/data/data_collection/live_detect/yanan_align_sort"
    fname = "/home/yaoqiang/data/tyres/20190108/dst_pics_full"

    for file_name in os.listdir(fname):
        file_path = os.path.join(fname, file_name)
        file_path = "/home/yaoqiang/Downloads/DCGAN-tensorflow/samplesEx"
        print file_path
        t = time()
        print 'start:'
        print t
        print RemoverRePic(file_path)
        print 'end:'
        print time() - t

