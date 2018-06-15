import cv2
import sys
import numpy as np 
import weed_detection as weed

cli_args = sys.argv[1:]

if len(cli_args) != 1:
    print("Usage: python video_feed.py video_path")
    sys.exit(1)

video_path = cli_args[0]

cap = cv2.VideoCapture(video_path)

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
