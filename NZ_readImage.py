import cv2
import numpy as np
import glob
from tqdm import tqdm

# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#
# print(flags)

temp = "spoof"
imagepaths = glob.glob("/home/yaoqiang/data/faceLiveness/" + temp + "/*.jpg")

path_len = len(imagepaths)

for i in tqdm(range(path_len)):
    imagelabel = imagepaths[i].split()
    # if (".jpg" != imagelabel[0][-4:]):
    #     print ("error format list %s" % imagelabel[0])
    #     continue

    # if i == 10:
    #     break
    print(imagelabel[0])

    img_oriXX = cv2.imread(imagelabel[0])
    img_ori = cv2.resize(img_oriXX, (400, 400), interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)
    img_yuv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YUV)
    img_YCrCb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YCrCb)

    htitch = np.hstack((img_ori, img_hsv, img_yuv, img_YCrCb))

    cv2.imshow(temp,htitch)
    # img2_ori = cv2.imread(imagelabel2[0])
    # img2_hsv = cv2.cvtColor(img2_ori, cv2.COLOR_BGR2HSV)
    # img2_yuv = cv2.cvtColor(img2_ori, cv2.COLOR_BGR2YUV)
    # img2_YCrCb = cv2.cvtColor(img2_ori, cv2.COLOR_BGR2YCrCb)
    #
    # htitch2 = np.hstack((img2_ori, img2_hsv, img2_yuv, img2_YCrCb))

    # vtitch = np.vstack((htitch, htitch2))


    # cv2.imshow(temp + "_img_ori", img_ori)
    # cv2.imshow(temp + "_img_hsv", img_hsv)
    # cv2.imshow(temp + "_img_yuv", img_yuv)
    # cv2.imshow(temp + "_img_YCrCb", img_YCrCb)
    #
    #
    #
    # cv2.imshow(temp2 + "_img2_ori", img2_ori)
    # cv2.imshow(temp2 + "_img2_hsv", img2_hsv)
    # cv2.imshow(temp2 + "_img2_yuv", img2_yuv)
    # cv2.imshow(temp2 + "_img2_YCrCb", img2_YCrCb)

    k=cv2.waitKey(0)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()
