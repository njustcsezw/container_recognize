import cv2
import numpy as np
from skimage.measure import label, regionprops
import os

import container_recognize.picture_hough_transform as hgt


def pro_process(img_path):
    #图片预处理

    #获取图片
    img = cv2.imread(img_path)
    #转换灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #图像二值化
    (_, thresh1) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    #霍夫变换(白底黑字)
    pic_hgt = hgt.picture_hough_transform(gray)
    #反转为检测连通域做准备(黑底白字) connected component region
    (_, hgt_ccr) = cv2.threshold(pic_hgt, 150, 255, cv2.THRESH_BINARY_INV)
    #cv2.imwrite(r'pictures/hougu.jpg', pic_hgt)

    #连通域检测
    (label_img, num_ccr) = label(hgt_ccr, return_num=True, connectivity=1)
    #print(num_ccr)
    props = regionprops(label_img)
    boxs = []

    for i in range(num_ccr):
        b = props[i].bbox
        #print(b)
        # b is tuple
        c = b[2] - b[0]
        d = b[3] - b[1]
        if 27 <= c <= 35 and 100 < b[0] < 320 and 200 < b[1] < 652:
            if 12 <= d <= 19:
                boxs.append(b)
        if 28 <= c <= 34 and 100 < b[0] < 320 and 200 < b[1] < 652:
            if 5 <= d <= 11:
                boxs.append(b)
        if 35 <= c <= 39 and 100 < b[0] < 320 and 200 < b[1] < 652:
            if 24 <= d <= 26:
                boxs.append(b)

    print(len(boxs))
    print(boxs)

    #清空即将使用的文件夹
    path = 'E:\\workpython2\\container_recognize\\test_sets'
    for i in os.listdir(path):
       path_file = os.path.join(path, i)
       if os.path.isfile(path_file):
          os.remove(path_file)
       else:
         for f in os.listdir(path_file):
             path_file2 =os.path.join(path_file, f)
             if os.path.isfile(path_file2):
                os.remove(path_file2)

    count_i = 1
    for box in boxs:
        x, y, w, h = box
        cv2.rectangle(img, (y, x), (h, w), (255, 0, 0), 2)
        copy_img = hgt_ccr[x:w, y:h]

        #fill the pics to 70*60 with black
        standard_copy_img = np.zeros([70, 60])
        height, width = copy_img.shape
        #print(width, height)
        x1 = int((70-height)/2)
        y1 = int((60-width)/2)
        standard_copy_img[x1:x1+height, y1:y1+width] = copy_img

        #save the cut of pictures
        save_path = r'test_sets/' + str(count_i) + '.jpg'
        cv2.imwrite(save_path, standard_copy_img)
        count_i += 1

    #cv2.imshow('original_img', img)
    #cv2.imshow('gray_img', gray)
    #cv2.imshow('binary_img', thresh1)
    #cv2.imshow('ccr_img', hgt_ccr)
    #cv2.imshow('hough_img', pic_hgt)

    #cv2.waitKey(0)
    #cv2.imwrite(save_path, crop_img)
    #cv2.destroyAllWindows()

#pro_process(r'pictures/a1.jpg')

