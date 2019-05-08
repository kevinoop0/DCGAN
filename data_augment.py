# -*- coding:utf-8 -*-
import os
import cv2
import math
import ipdb
def rotate(img,angle):
    height = img.shape[0]
    width = img.shape[1]
    if angle%180 == 0:
        scale = 1
    elif angle%90 == 0:
        scale = float(max(height, width))/min(height, width)
    else:
        scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)

    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    return rotateImg

data_path='./data/faces/'
imgs= os.listdir(data_path)

for i,img in enumerate(imgs,1):
    # ipdb.set_trace()
    img_path= os.path.join(data_path,img)
    image= cv2.imread(img_path)
    image_rot45= rotate(image,45)
    image_rot315 = rotate(image, -45)
    image_flip = cv2.flip(image,1)
    fix_name = './data/face/{}'.format(i)
    cv2.imwrite(fix_name+'_rot45'+'.jpg', image_rot45)
    cv2.imwrite(fix_name+'_rot315'+'.jpg', image_rot315)
    cv2.imwrite(fix_name+'_flip'+'.jpg', image_flip)
