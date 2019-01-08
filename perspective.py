import cv2
import numpy as np


#获取图片
img_path = r'pictures/a5.jpg'
img = cv2.imread(img_path)
rows, cols, ch = img.shape
print(rows, cols, ch)

pts1 = np.float32([[0, 0], [700, 0], [70, 520], [650, 525]])
pts2 = np.float32([[0, 0], [400, 0], [0, 500], [400, 500]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (350, 400))

cv2.imshow("img", img)
cv2.imshow("per_img", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
