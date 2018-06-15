'''
Python 3.6.4
OpenCV 3.4.0
Numpy 1.14.0
'''
import cv2
import numpy as np 
import weed_detection as weed

cap = cv2.VideoCapture('input_video.mp4')

_, first_frame = cap.read() 

wd = weed.WeedDetection(first_frame)

while(True):
    ret,frame = cap.read()
    if ret is False:
        break
    
    frame_hsv = wd.preprocess(frame)
    
    msk = wd.createMask(frame_hsv)
    
    msk = wd.transform(msk)

    percentage = wd.weedPercentage(msk)
    
    res = wd.markPercentage(frame.copy(), percentage)
    bit_msk = cv2.bitwise_and(frame,frame,mask = msk)

    cv2.imshow('FEED', frame)
    cv2.imshow('Result', res)
    cv2.imshow('Bit Mask', bit_msk)

    k = cv2.waitKey(2)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
