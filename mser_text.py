import cv2


#获取图片
img_path = r'pictures/a1.jpg'
img = cv2.imread(img_path)

#灰度化，高斯滤波
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
#canny = cv2.Canny(blurred, 50, 150) //效果很差

'''
#提取图像的梯度
#以Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

#继续去噪声,且图像二值化
blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
'''

#
mser = cv2.MSER_create(_min_area=400, _max_area=1200)
regions, boxes = mser.detectRegions(blurred)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#cv2.imwrite(save_path, crop_img)
cv2.imshow('blurred', blurred)
cv2.imshow("mser_img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

