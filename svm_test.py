from svmutil import *  # libSVM
from svm import *  # libSVM
import glob
import cv2
import numpy as np
import random
from multiprocessing import Process
import sys

g_width = 50
g_height = 50

def getPicsTxt(tag):
    if tag == 'train':
        root_path = '/home/yaoqiang/SIW/Test/001-live-select/'
        dstFileName = './test.txt'
    else:
        root_path = '/home/yaoqiang/SIW/Temp/'
        dstFileName = './validation.txt'
    image_paths = glob.glob(root_path + "*.jpg")
    random.shuffle(image_paths)

    path_len = len(image_paths)
    # print(path_len)

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

def GetImgAndLabelFromTxt(tag):
    if tag == 'train':
        dstFileName = './train_random.txt'
    else:
        dstFileName = './validation.txt'
    dst_path = "/home/yaoqiang/SIW/" + dstFileName

    txt_paths = glob.glob(dst_path)
    path_len = len(txt_paths)
    img_set = []
    label_set = []
    img_width = g_width
    img_height = g_height
    rescale = 1.0
    for i in (range(path_len)):
        txt_path = txt_paths[i]
        fd = open(txt_path, 'r')
        for line in fd:
            line = line.strip()
            img_path, label = line.split(" ")
            # print(img_path)
            # print(label)
            img = cv2.imread(img_path)
            if img is None:
                print(img_path)

            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
            imgX = cv2.resize(img, (img_width, img_height))
            # img_hsv = imgX
            img_hsv = cv2.cvtColor(imgX, cv2.COLOR_BGR2HSV)
            cv2.imshow('Video', img_hsv)
            c = cv2.waitKey(10)
            if c == 27:
                break
            # get row vector
            feature = img_hsv.reshape(-1)
            # print(feature.shape)
            img_set.append(feature)
            label_set.append(label)
        fd.close()
    np_img = np.array(img_set)*rescale
    np_label = np.array(label_set, dtype = np.int)
    # np_label = np_utils.to_categorical(np_label, num_classes=2)

    # print(np_img.shape)
    # print(np_label)
    # print(np_label.shape)
    # print(np_label[0:10, :])
    # print(np_label.dtype)
    # print(np_img.shape)
    # print(np_img.dtype)
    return np_label, np_img

def classify(model, labels, imgs):
    count = 0
    correct = 0
    for i in range(len(labels)):
        p_label, p_acc, p_val = svm_predict([0], [imgs[i]], model, '-q')
        if p_label[0] == labels[i]:
            correct += 1
        count += 1
    print ("%d %d %f" % (correct, count, (float(correct) / count)))
    return

def GeImgFeature(img):
    # img = cv2.imread(path)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    imgX = cv2.resize(img, (g_width, g_height))
    img_hsv = cv2.cvtColor(imgX, cv2.COLOR_BGR2HSV)
    # cv2.imshow('test', img_hsv)
    # c = cv2.waitKey(0)
    feature = img_hsv.reshape(-1)
    return feature

def GetVideoNegSamplesBySVM(model, video_paths, DstPath, thread, label):
    path_len = len(video_paths)
    k = 500*thread
    for item in range(path_len):
        cap = cv2.VideoCapture(video_paths[item])
        fd = open(video_paths[item][:-4] + '.face')
        count = 0
        correct = 0
        while True:
            line = fd.readline()
            if not line:
                break
            # print(line)
            w1, h1, w2, h2 = line.split()
            w1 = int(w1)
            w2 = int(w2)
            h1 = int(h1)
            h2 = int(h2)
            ret, frame = cap.read()
            if frame is None:
                break
            if h1 <= 0:
                continue
            if w1 <= 0:
                continue
            if h2 <= 0:
                continue
            if w2 <= 0:
                continue
            if h1 >= h2:
                print (sys._getframe().f_lineno)
                continue
            if w1 >= w2:
                print (sys._getframe().f_lineno)
                continue
            # frame = frame[310:800, 630:1120]
            frameX = frame[int(h1):int(h2), int(w1):int(w2)]
            feature = GeImgFeature(frameX)
            p_label, p_acc, p_val = svm_predict([0], [feature], model, '-q')

            count += 1
            if label == p_label[0]:
                correct += 1
            else:
                k += 1
                cv2.imwrite(DstPath + bytes(k) + '.jpg', frameX)

            cv2.rectangle(frame, (int(w1), int(h1)), (int(w2), int(h2)), (0, 255, 0), 2)
            cv2.imshow('Video', frame)

            # cv2.imwrite(DstImagepaths + "_" + bytes(i) + '.jpg', frame)
            # i = i + 1
            c = cv2.waitKey(100)
            if c == 27:
                break
        print ("%s %d %d %f" % (video_paths[item], correct, count, (float(correct) / count)))
        cap.release()
        cv2.destroyAllWindows()
        fd.close()
    return

def train_svm_model():
    # getPicsTxt('train')
    label_total, img_total = GetImgAndLabelFromTxt('train')
    print(len(label_total))
    print(label_total[:20])
    testSplit = int(.7 * len(label_total))

    label_train = label_total[:testSplit]
    label_test = label_total[testSplit:]

    img_train = img_total[:testSplit]
    img_test = img_total[testSplit:]

    # train
    prob = svm_problem(label_train, img_train)
    param = svm_parameter('-q')
    param.probability = 1
    param.kernel_type = LINEAR
    param.C = 1
    model = svm_train(prob, param)
    svm_save_model('live_model_file', model)

    # test
    classify(model, label_train, img_train)
    classify(model, label_test, img_test)

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

def thread_testVideo():
    model = svm_load_model('live_model_file')
    video_path = "/darray_det/liveness/SiW_release/Test/spoof/001/001-1-2-1-1.mov"
    DstPath = "/home/yaoqiang/SIW/Test/001-spoof/"
    threads = 1

    video_paths = glob.glob(video_path)
    print(video_paths)
    div_lists = _div_list(video_paths, threads)
    func = GetVideoNegSamplesBySVM
    task = []
    for i in xrange(threads):
        t = Process(target=func, args=(model, div_lists[i], DstPath, i, 2))
        task.append(t)

    for i in xrange(threads):
        #        print ("start threads :" + str(i))
        task[i].start()

    for i in xrange(threads):
        #        print ("over threads :" + str(i))
        task[i].join()
    return

if __name__ == "__main__":
    # getPicsTxt('train')
    # train_svm_model()
    thread_testVideo()


# y, x = [1,-1], [{1:1, 2:1}, {1:-1,2:-1}]
# prob = svm_problem(y, x)
#
# param = svm_parameter('-t 0 -c 4 -b 1')
# model = svm_train(prob, param)
# yt = [1]
# xt = [{1:1, 2:1}]
# model = svm_load_model('model_file')
# p_label, p_acc, p_val = svm_predict(yt, xt, model)

# svm_save_model('model_file', model)
