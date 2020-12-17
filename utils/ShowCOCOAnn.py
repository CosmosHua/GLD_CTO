#!/usr/bin/python3
# coding: utf-8

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import matplotlib; matplotlib.use("qt5agg")


#######################################################
def ShowAnn(root):
    imgDir = root+"/JPEGImages"
    annFile = root+"/annotations.json"
    coco=COCO(annFile) # COCOAPI load/init annotations

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [i['name'] for i in cats] # sorted
    print('\nCOCO categories: \n{}\n'.format(', '.join(nms)))
    nms = set([i['supercategory'] for i in cats])
    if None in nms: nms.discard(None); nms.add("NULL")
    print('COCO supercategories: \n{}'.format(', '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=["door_glass"])
    imgIds = coco.getImgIds(catIds=catIds)
    print("\nImages' IDs: ", imgIds)
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    print("Selected Image's INFO: ", img)

    # load and display image
    if "JPEGImages" in img['file_name']: imgDir = root
    I = io.imread('%s/%s'%(imgDir, img['file_name']))
    #plt.imshow(I); plt.axis('off'); plt.show()

    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns); plt.show()


#######################################################
if __name__ == "__main__":
    ShowAnn('coco_door')
