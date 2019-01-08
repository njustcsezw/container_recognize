import cv2
import numpy as np
from skimage.measure import label, regionprops


import container_recognize.picture_hough_transform as hgt

#获取图片
img_path = r'pictures/a1.jpg'
img = cv2.imread(img_path)

#转换灰度
#去噪有很多种方法，均值滤波器、高斯滤波器、中值滤波器、双边滤波器等
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#print(gray_hist)

#图像二值化
(_, thresh1) = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
(_, thresh2) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

#霍夫变换(白底黑字)
hgt = hgt.picture_hough_transform(gray)
#反转为检测连通域做准备(黑底白字) connected component region
(_, hgt_ccr) = cv2.threshold(hgt, 120, 255, cv2.THRESH_BINARY_INV)
#(_, hgt2bgr) = cv2.threshold(hgt, 150, 255, cv2.COLOR_GRAY2BGR)
#cv2.imwrite(r'pictures/hougu.jpg', hgt)

#连通域检测
(label_img, num_ccr) = label(hgt_ccr, return_num=True, connectivity=1)
print(num_ccr)
props = regionprops(label_img)
boxs = []

'''
crop_img = []
b = props[72].bbox
crop_img = hgt[b[0]:b[2], b[1]:b[3]]
cv2.imshow('crop_img', crop_img)

crop_img1 = []
b = props[77].bbox
crop_img_1 = hgt[b[0]:b[2], b[1]:b[3]]
cv2.imshow('crop_img1', crop_img_1)

crop_img2 = []
b = props[32].bbox
crop_img2 = hgt[b[0]:b[2], b[1]:b[3]]
cv2.imshow('crop_img2', crop_img2)
'''

for i in range(num_ccr):
    b = props[i].bbox
    # b is tuple
    print(b)
    c = b[2] - b[0]
    d = b[3] - b[1]
    if 18 <= c <= 27:
        if 8 <= d <= 16:
            boxs.append(b)
    if 18 <= c <= 27:
        if 5 <= d <= 6:
            boxs.append(b)


print(len(boxs))
print(boxs)

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

cv2.imshow('original_img', img)
#cv2.imshow('gray_img', gray)
#cv2.imshow('binary_img', thresh2)
#cv2.imshow('ccr_img', hgt2bgr)
cv2.imshow('hough_img', hgt)

cv2.waitKey(0)
#cv2.imwrite(save_path, crop_img)
cv2.destroyAllWindows()
