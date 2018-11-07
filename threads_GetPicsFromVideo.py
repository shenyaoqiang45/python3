import glob
import cv2
import sys
import random
from multiprocessing import Process

def GetPicsFromVideo():
    class_name = "live"
    movie_paths = glob.glob("/darray_det/liveness/SiW_release/Train/live/*/*.mov")
    DstImagepaths = "/home/yaoqiang/SIW/Train2/" + class_name + "/"
    path_len = len(movie_paths)
    fps = 600
    j = 0
    for i in range(path_len):
        # print(movie_paths[i])
        movie_path = movie_paths[i]
        file_path = movie_path[:-3] + 'face'
        print(movie_path)
        print(file_path)
        cap = cv2.VideoCapture(movie_path)
        fd = open(file_path)
        c = 0
        s = 0
        while True:
            line = fd.readline()
            if not line:
                break
            w1, h1, w2, h2 = line.split()
            w1 = int(w1)
            w2 = int(w2)
            h1 = int(h1)
            h2 = int(h2)
            ret, frame = cap.read()
            if h1 == 0:
                print (sys._getframe().f_lineno)
                continue
            if w1 == 0:
                print (sys._getframe().f_lineno)
                continue
            if h2 == 0:
                print (sys._getframe().f_lineno)
                continue
            if w2 == 0:
                print (sys._getframe().f_lineno)
                continue
            if s % fps == 0:
                frame = frame[h1:h2, w1:w2]
                # cv2.imshow('Video', frame)
                # c = cv2.waitKey(100)
                # if c == 27:
                #     break
                cv2.imwrite(DstImagepaths + class_name + "_" + bytes(j) + '.jpg', frame)
                j = j + 1
                s = s + 1
            if c == 27:
                break
        cap.release()
        # cv2.destroyAllWindows()
        fd.close()
        if c == 27:
            break
    return

def getPicsTxt(tag):
    if tag == 'train':
        root_path = "/home/yaoqiang/SIW/Train2/*/"
        dstFileName = './train_random.txt'
    else:
        root_path = "/home/yaoqiang/PycharmProjects/kerasTUT/data/validation/*/"
        dstFileName = './validation.txt'
    image_paths = glob.glob(root_path + "*.jpg")
    random.shuffle(image_paths)
    path_len = len(image_paths)
    print(path_len)

    dst_path = "/home/yaoqiang/SIW/"
    dstFile = dst_path + dstFileName

    fd = open(dstFile, 'w')
    for i in (range(path_len)):
        imagepath = image_paths[i]
        if 'live' in imagepath:
            fd.write(imagepath + ' ' + '1\n', )
        else:
            fd.write(imagepath + ' ' + '2\n', )
    fd.close()
    return

def _div_list(ls,n):
    if not isinstance(ls,list) or not isinstance(n,int):
        print "type error exit!!!"
        return []
    ls_len = len(ls)
    if n<=0 or 0==ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len/n
        k = ls_len%n

        ls_return = []
        for i in xrange(0,(n-1)*j,j):
            ls_return.append(ls[i:i+j])


        ls_return.append(ls[(n-1)*j:])
        return ls_return


def threads_GetPicsFromVideo(movie_paths, DstImagepaths, indics):
    path_len = len(movie_paths)
    print(path_len)
    fps = 300
    j = 0
    for i in range(path_len):
        # print(movie_paths[i])
        movie_path = movie_paths[i]
        file_path = movie_path[:-3] + 'face'
        print(movie_path)
        print(file_path)
        cap = cv2.VideoCapture(movie_path)
        fd = open(file_path)
        c = 0
        s = 0
        while True:
            line = fd.readline()
            if not line:
                break
            w1, h1, w2, h2 = line.split()
            w1 = int(w1)
            w2 = int(w2)
            h1 = int(h1)
            h2 = int(h2)
            ret, frame = cap.read()
            if h1 == 0:
                print (sys._getframe().f_lineno)
                continue
            if w1 == 0:
                print (sys._getframe().f_lineno)
                continue
            if h2 == 0:
                print (sys._getframe().f_lineno)
                continue
            if w2 == 0:
                print (sys._getframe().f_lineno)
                continue
            if s % fps == 0:
                frame = frame[h1:h2, w1:w2]
                # cv2.imshow('Video', frame)
                # c = cv2.waitKey(100)
                # if c == 27:
                #     break
                k = 360 * indics + j
                cv2.imwrite(DstImagepaths + "spoof_" + bytes(k) + '.jpg', frame)
                j = j + 1
                s = s + 1
            if c == 27:
                break
        cap.release()
        # cv2.destroyAllWindows()
        fd.close()
        if c == 27:
            break
    return

if __name__ == "__main__":
    # GetPicsFromVideo()
    # getPicsTxt('train')
    movie_paths = glob.glob("/darray_det/liveness/SiW_release/Train/spoof/*/*.mov")
    Image_paths = "/home/yaoqiang/SIW/Train2/spoof/"
    threads = 4

    div_lists = _div_list(movie_paths, threads)
    func = threads_GetPicsFromVideo
    task = []
    for i in xrange(threads):
        t = Process(target=func, args=(div_lists[i], Image_paths, i))
        task.append(t)

    for i in xrange(threads):
        #        print ("start threads :" + str(i))
        task[i].start()

    for i in xrange(threads):
        #        print ("over threads :" + str(i))
        task[i].join()
