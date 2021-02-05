#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, torch
from time import time
from PIL import Image
# pip3 install facenet-pytorch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets

post_process = lambda x: (x-127.5)/128 # for faces tensor
T2N = lambda x: x.permute(0,2,3,1).numpy().astype('uint8')[...,::-1]
##########################################################################################
def load_img(im, ratio=1): # im=BGR, img=RGB
    if type(im)==str and os.path.isfile(im): im = cv2.imread(im)
    if type(im)==np.ndarray: img = Image.fromarray(im[...,::-1])
    if type(im)==Image.Image: img, im = im, np.array(im)[...,::-1]
    return im, img.resize((np.array(img.size)*ratio).astype(int))
    #return im, img.resize([int(i*ratio) for i in img.size])


def cat_face(faces):
    faces = T2N(faces); n,h,w,c = faces.shape
    dh = int(n**0.5); dw = int(np.ceil(n/dh))
    if dh>dw: dh, dw = dw, dh # aim: dw>=dh
    im = np.zeros((dh*(h+1)+1, dw*(w+1)+1, c))
    for n,fc in enumerate(faces):
        i,j = 1+(n//dw)*(h+1), 1+(n%dw)*(w+1)
        im[i:i+h, j:j+w] = fc
    return im.astype('uint8')


def face_det(im, mtcnn, resnet=None, ratio=0.5, show=0):
    if type(mtcnn)!=MTCNN: mtcnn = MTCNN()
    mtcnn.factor = 0.6; mtcnn.keep_all = True
    mtcnn.margin = 14; mtcnn.select_largest = False
    mtcnn.post_process = False; bc = (0,255,255)
    
    im, img = load_img(im, ratio); dt = time();
    boxes, probs = mtcnn.detect(img, landmarks=False)
    faces = mtcnn.extract(img, boxes, save_path=None)
    #faces, boxes, probs = mtcnn(img, landmarks=False)
    dt = (time()-dt)*1E3; # mtcnn.forward is modified
    if boxes is None: return im, boxes, probs, faces

    '''if type(resnet)!=InceptionResnetV1:
        if type(resnet)!=str or not os.path.isfile(resnet):
            resnet = 'yolov5/vggface2.pt'
        resnet = InceptionResnetV1(pretrained=resnet).eval()'''

    for b, p in zip(boxes, probs):
        b = (b/ratio).astype(int); info = '%.3f'%p
        r = min(2,(b[2]-b[0])/30); h = b[1]-1-int(9*r)
        cv2.rectangle(im, (b[0],b[1]), (b[2],b[3]), bc)
        cv2.rectangle(im, (b[0],h), (b[2],b[1]), bc, -1)
        cv2.putText(im, info, (b[0],b[1]-1), 3, 0.3*r, (0,)*3)

    if show>0: print('%.2fms:'%dt, probs)
    if show>1: cv2.imshow('det', im)
    if show>2: cv2.imshow('face', cat_face(faces))
    return im, boxes, probs, faces


##########################################################################################
if __name__ == '__main__':
    im = 'openpose/images/ski.jpg'
    im = face_det(im, MTCNN(), show=3)[0]
    cv2.waitKey(); cv2.destroyAllWindows()

