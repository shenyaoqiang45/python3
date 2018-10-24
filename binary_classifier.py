import glob
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# data pre-processing 1
def getPicsTxt(tag):
    if tag == 'train':
        root_path = "/home/yaoqiang/PycharmProjects/kerasTUT/data/train/*/"
        dstFileName = './train.txt'
    else:
        root_path = "/home/yaoqiang/PycharmProjects/kerasTUT/data/validation/*/"
        dstFileName = './validation.txt'
    image_paths = glob.glob(root_path + "*.jpg")
    path_len = len(image_paths)
    print(path_len)

    dst_path = "/home/yaoqiang/PycharmProjects/kerasTUT/liveness_txt/"
    dstFile = dst_path + dstFileName

    fd = open(dstFile, 'w')
    for i in (range(path_len)):
        imagepath = image_paths[i]
        if 'real' in imagepath:
            fd.write(imagepath + ' ' + '1\n', )
        else:
            fd.write(imagepath + ' ' + '0\n', )
    fd.close()
    return


# data pre-processing 2
def GetImgAndLabelFromTxt(tag):
    if tag == 'train':
        dstFileName = './train.txt'
    else:
        dstFileName = './validation.txt'
    dst_path = "/home/yaoqiang/PycharmProjects/kerasTUT/liveness_txt/" + dstFileName

    txt_paths = glob.glob(dst_path)
    path_len = len(txt_paths)
    img_set = []
    label_set = []
    img_width = 150
    img_height = 150
    rescale = 1.0/255
    # random.shuffle(txt_paths)
    for i in (range(path_len)):
        txt_path = txt_paths[i]
        fd = open(txt_path, 'r')
        for line in fd:
            line = line.strip()
            img_path, label = line.split(" ")
            # print(img_path)
            # print(label)
            img = cv2.imread(img_path)
            # cv2.imshow('Video', img)
            # c = cv2.waitKey(100)
            # if c == 27:
            #     break
            imgX = cv2.resize(img, (img_width, img_height))
            img_set.append(imgX)
            label_set.append(label)
        fd.close()

    np_img = np.array(img_set)*rescale
    np_label = np.array(label_set, dtype = np.int)
    np_label = np_utils.to_categorical(np_label, num_classes=2)
    # print(np_label[0:10, :])
    # print(np_label.dtype)
    # print(np_img.shape)
    # print(np_img.dtype)
    return np_img, np_label


# build your neural net
def AlexNet(img_width=150, img_height=150):
    print(K.image_data_format())
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2)) # binary_calssifier
    model.add(Activation('softmax'))

    return model

# define your optimizer

# train the model

# test

# predict

# main
if __name__ == "__main__":
    global_img_width = 150
    global_img_height = 150
    rescale = 1.0/255
    epochs = 10
    batch_size = 20

    # data pre-processing
    getPicsTxt('train')
    getPicsTxt('validastion')
    X_train, Y_train = GetImgAndLabelFromTxt('train')
    X_validation, Y_validation = GetImgAndLabelFromTxt('validastion')
    # print(X_validation.shape)
    # print(Y_validation.shape)

    # build your neural net
    model = AlexNet(img_width=global_img_width, img_height=global_img_height)

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # define your optimizer
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model
    # print('Training ------------')
    # model.fit(X_train, Y_train, epochs=10, batch_size=20)
    # model.save_weights('syq_model_weights.h5')
    # model.load_weights('syq_model_weights.h5')

    # Validation
    # train_log = model.fit(X_train, Y_train,
    #                       batch_size=batch_size, nb_epoch=epochs,
    #                       validation_data=(X_validation, Y_validation))
    # model.save_weights('syq_model_weights.h5')
    model.load_weights('syq_model_weights.h5')

    # matplot
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy on sar classifier")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="upper right")
    # plt.savefig("Loss_Accuracy_alexnet_syq_{:d}e_{:d}batch.jpg".format(epochs, batch_size))


    # Testing
    # print('\nTesting ------------')
    # print(X_train[1:2, :].shape)
    # y_temp = model.predict(X_train[1:2, :])
    # print(y_temp)
    # sum = y_temp[0][0] + y_temp[0][1]
    # print(sum)
    # print(Y_train[1:2, :])
    # print(np.argmax(y_temp))


    # visualize from image
    # imagepaths = glob.glob("/home/yaoqiang/PycharmProjects/kerasTUT/data/validation/spoof/*.jpg")
    # path_len = len(imagepaths)
    #
    # errNum = 0
    # for i in tqdm(range(path_len)):
    #     imagelabel = imagepaths[i].split()
    #     img_ori = cv2.imread(imagelabel[0])
    #     dst_frame = cv2.resize(img_ori, (global_img_width, global_img_height))
    #     x_input = dst_frame.reshape(-1, dst_frame.shape[0], dst_frame.shape[1], dst_frame.shape[2])
    #     # x_input = x_input * rescale
    #     np_x_imput = np.array(x_input) * rescale
    #     y_pre = model.predict(np_x_imput)
    #     # print(y_pre.dtype)
    #     print("{:.2f}, {:.2f}".format(y_pre[0][0], y_pre[0][1]))
    #     if np.argmax(y_pre) == 1:
    #         errNum = errNum + 1
    #         # cv2.imshow('Video', dst_frame)
    #         # c = cv2.waitKey(2000)
    #         # if c == 27:
    #         #     break
    # # cv2.destroyAllWindows()
    #
    # print(errNum)
    # print(path_len)


    # visualize from video
    img_width = global_img_width
    img_height = global_img_height

    # video_path = "/darray_det/liveness/SiW_release/Train/live/006/006-1-1-1-1.mov"
    # video_path = "/darray_det/liveness/SiW_release/Train/spoof/115/115-1-2-1-1.mov"
    # video_path = "/darray_det/liveness/SiW_release/Test/spoof/001/001-1-2-1-1.mov"
    # video_path = "/darray_det/liveness/SiW_release/Test/spoof/158/158-1-3-1-1.mov"
    video_path = "/darray_det/liveness/SiW_release/Test/spoof/057/057-1-2-1-1.mov"
    cap = cv2.VideoCapture(video_path)
    fd = open(video_path[:-4] + '.face')
    # DstImagepaths = "/home/yaoqiang/data/SIW/"
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        line = fd.readline()
        if not line:
            break
        # print(line)
        w1, h1, w2, h2 = line.split()
        w1 = int(w1)
        h1 = int(h1)
        w2 = int(w2)
        h2 = int(h2)

        ret, frame = cap.read()
        if w1 == 0:
            continue
        # frame = frame[310:800, 630:1120]
        # frameX = frame[int(h1):int(h2), int(w1):int(w2)]

        frameX = frame[h1:h2, w1:w2]

        dst_frame = cv2.resize(frameX, (img_width, img_height))


        # print(dst_frame.shape)
        # print(dst_frame.ndim)
        # print(type(dst_frame))

        x_input = dst_frame.reshape(-1, dst_frame.shape[0], dst_frame.shape[1], dst_frame.shape[2])
        # print(type(x_input))
        x_input = x_input * rescale
        # print(x_input.shape)
        # print(x_input.ndim)
        # print(type(x_input))
        y_pre = model.predict(x_input)
        print("{:.2f}, {:.2f}".format(y_pre[0][0], y_pre[0][1]))
        # print(y_pre)
        cv2.rectangle(frame, (w1, h1), (w2, h2), (0, 255, 0), 2)
        if np.argmax(y_pre) == 1:
            testStr = 'live'
        else:
            testStr = 'spoof'
        cv2.putText(frame, testStr, (w1-10, h1-10), font, 1.2, (0, 255, 0), 2)
        cv2.imshow('Video', frame)

        # cv2.imwrite(DstImagepaths + "_" + bytes(i) + '.jpg', frame)
        # i = i + 1
        c = cv2.waitKey(100)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    fd.close()