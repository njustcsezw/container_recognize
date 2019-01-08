import cv2
import numpy as np

#获取图片
img_path = r'pictures/a1.jpg'
img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Canny只能处理灰度图
#ml模块
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#高斯平滑处理原图像降噪
canny = cv2.Canny(blurred, 50, 150)
#apertureSize默认为3
#cv2.imwrite(r'pictures/a111.jpg', canny)
#(_, thresh) = cv2.threshold(canny, 90, 255, cv2.THRESH_BINARY)

cv2.imshow('Original', img)
cv2.imshow('Canny', canny)
#cv2.imshow('Thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
