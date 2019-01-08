import cv2
import numpy as np
from matplotlib import pyplot as plt

#获取图片
img_path = r'pictures/a6.jpg'
img = cv2.imread(img_path)

#转换灰度并去噪声
#去噪有很多种方法，均值滤波器、高斯滤波器、中值滤波器、双边滤波器等
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#图像二值化
(_, thresh1) = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
(_, thresh2) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
(_, thresh3) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
(_, thresh4) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)


# 统计概率霍夫线变换
'''
参数1：要检测的二值图（一般是阈值分割或边缘检测后的图）
参数2：距离r的精度，值越大，考虑越多的线
参数3：角度θ的精度，值越小，考虑越多的线
参数4：累加数阈值，值越小，考虑越多的线  
minLineLength：最短长度阈值，比这个长度短的线会被排除
maxLineGap：同一直线两点之间的最大距离 
'''
drawing = np.zeros(img.shape[:], dtype=np.uint8)
lines = cv2.HoughLinesP(thresh2, 0.8, np.pi / 180, 90,
                        minLineLength=40, maxLineGap=1)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(thresh3, (x1, y1), (x2, y2), 255, 6, lineType=cv2.LINE_AA)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 6, lineType=cv2.LINE_AA)

#cv2.imshow('original_img', img)
#cv2.imshow('gray_img', gray)

cv2.imshow('binary_img', thresh4)
cv2.imshow('hough_img', thresh3)

cv2.imshow('drawing', drawing)
cv2.waitKey(0)
#cv2.imwrite(r'pictures/hougu.jpg', thresh3)
cv2.destroyAllWindows()
