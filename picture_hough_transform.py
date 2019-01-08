import cv2
import numpy as np


def picture_hough_transform(gray):
    (_, thresh) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    (_, thresh1) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(thresh, 0.8, np.pi / 180, 90,
                            minLineLength=50, maxLineGap=1)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(thresh1, (x1, y1), (x2, y2), 255, 6, lineType=cv2.LINE_AA)
    return thresh1


def picture_hough_transform1(gray):
    (_, thresh) = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    (_, thresh1) = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(thresh, 0.8, np.pi / 180, 90,
                            minLineLength=55, maxLineGap=1)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(thresh1, (x1, y1), (x2, y2), 255, 6, lineType=cv2.LINE_AA)
    return thresh1
