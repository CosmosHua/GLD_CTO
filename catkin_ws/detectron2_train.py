#!/usr/bin/python3
# coding: utf-8

import os, cv2, random, yaml
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import Instances, Boxes
from time import time; from glob import glob
from torch import empty; import numpy as np
from tqdm import tqdm, trange


#######################################################
def view_ann(data_name):
    dicts = DatasetCatalog.get(data_name) # dicts[2]
    metadata = MetadataCatalog.get(data_name) # metadata
    # visualize the annotations
    for i in random.sample(dicts, 100):
        im = cv2.imread(i['file_name'])[:,:,::-1]
        vis = Visualizer(im, metadata=metadata, scale=0.6)
        im = vis.draw_dataset_dict(i).get_image()
        cv2.imshow('im', im[:,:,::-1])
        if cv2.waitKey(0)==27: break
    cv2.destroyAllWindows()


#######################################################
def Setup2Train(data_dir, ym, N=500):
    data_name = 'doors'; json = data_dir+'/annotations.json'
    register_coco_instances(data_name, {}, json, data_dir)
    dicts = DatasetCatalog.get(data_name) # dicts[2]
    metadata = MetadataCatalog.get(data_name) # metadata
    metadata.thing_colors = [tuple(np.random.randint(0,256,3)) #list
        for c in metadata.thing_classes] #random.choices(range(256),k=3) 

    cfg = get_cfg(); cfg.merge_from_file(ym)
    cfg.MODEL.WEIGHTS = ym.replace('.yaml','.pkl')
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    cfg.DATASETS.TRAIN = (data_name,)
    cfg.DATASETS.TEST = () # no metrics implemented yet
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = N
    cfg.OUTPUT_DIR = data_dir+'/models'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(ym, 'rb') as f: dt = yaml.load(f)
    ym_base = os.path.join(os.path.dirname(ym), dt['_BASE_'])
    os.system(' '.join(['cp', ym_base, cfg.OUTPUT_DIR]))
    dt['_BASE_'] = os.path.basename(dt['_BASE_'])
    ym_ = ym.replace(os.path.dirname(ym), cfg.OUTPUT_DIR)
    with open(ym_, 'w+') as f: yaml.dump(dt, f, indent=4)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def Setup4Infer(data_dir, thd=0.6):
    data_name = 'doors'; json = data_dir+'/annotations.json'
    register_coco_instances(data_name, {}, json, data_dir)
    dicts = DatasetCatalog.get(data_name) # dicts[2]
    metadata = MetadataCatalog.get(data_name) # metadata

    model = glob(data_dir+'/models/*.pth')[-1]
    ym = glob(os.path.dirname(model)+'/*x.yaml')[0]
    assert os.path.isfile(model) and os.path.isfile(ym)
    print('Loading model: %s' % model)

    cfg = get_cfg(); cfg.merge_from_file(ym)
    cfg.DATASETS.TEST = (data_name,)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thd # test threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    return DefaultPredictor(cfg), metadata


RES = lambda x,s: cv2.resize(x, NewSZ(x,s))
INT = lambda x,f=1: tuple([int(i*f) for i in x]) # round
RSZ = lambda x,s: cv2.resize(x,s) if type(s)==tuple else cv2.resize(x,None,fx=s,fy=s)
#######################################################
def NewSZ(wh, sz=1.0):
    if type(wh)==cv2.UMat: wh = wh.get()
    if type(wh)==np.ndarray: wh = wh.shape[1::-1]
    else: assert type(wh) in (tuple, list)
    if type(sz)==float: sz = INT(wh, sz)
    elif type(sz)==int: sz = INT(wh, sz/wh[1])
    return sz if type(sz)==tuple else tuple(wh)


def Infer2Img(im, pred, meta, sz=1.0, info=''): # BGR
    t = time(); old = out = pred(im)['instances'].to('cpu')
    out = Sift_CLS(out, meta, [0,1,2,'door_unknown'], 'include')
    out = Sift_BOX(out, meta, wt=weight); sz = NewSZ(im,sz)
    if len(old)!=len(out): print(old.pred_classes,'->',out.pred_classes)
    vis = Visualizer(im[:,:,::-1], metadata=meta, scale=max(1,720/im.shape[0]),
                    instance_mode=ColorMode.SEGMENTATION)
    im = vis.draw_instance_predictions(out).get_image()[:,:,::-1]
    t = time()-t; h,w = im.shape[:2]; im = cv2.UMat(im) # OpenCV 4.2+
    cv2.putText(im, '%.1fms'%(t*1000), (w-75,h-8), 4, 0.5, (0,255,255), 1)
    cv2.putText(im, info, (5,h-8), 4, 0.5, (0,255,255), 1)
    return cv2.resize(im.get(),sz), out


weight = dict(door_way=1, door_open=1, door_close=1)
#######################################################
def Sift_BOX(out, meta, thd=10, wt=None):
    if len(out)<0: return out # unnecessary
    ct = out.pred_boxes.get_centers(); id = []
    pred = out.pred_classes; prob = out.scores
    if type(wt)==dict:
        cls = meta.thing_classes; idx = cls.index
        wt = {idx(k):wt[k] for k in wt if k in cls} # map->id
        prob = prob + sum((pred==k)*wt[k] for k in wt) # adjust
        '''wt = [wt[i] if cls_na[i] in wt else 0 for i in pred]
        prob = prob + np.array(wt) # adjust prob'''
    for c in ct: # (N,2)
        dis = np.linalg.norm(ct-c, axis=1) # distance
        p = np.where(dis<thd)[0] # np.nonzero(dis<thd)[0]
        k = p[prob[p].sort(0,True).indices[0]] # sort
        if k not in id: id.append(k)
    return out[id]


def Sift_CLS(out, meta, cls=[], mod='ex'):
    if type(cls) not in (list,tuple): cls = [cls]
    cls_na = meta.thing_classes; id = cls_na.index
    cls_id = meta.thing_dataset_id_to_contiguous_id
    pred = out.pred_classes; h,w = out.image_size; idx = []
    for c in cls: # filter and map_>id
        if type(c)==str and c in cls_na: idx += [id(c)]
        elif type(c)==int and c in cls_id: idx += [c]
    if len(out)<0 or len(idx)<len(cls): return out

    if 'in' in mod: idx = [i for i,d in enumerate(pred) if d in idx]
    else: idx = [i for i,d in enumerate(pred) if d not in idx]
    if len(idx)>0: return out[idx] # out.__class__.cat()
    new = dict(pred_boxes=Boxes(empty(0,4)), scores=empty(0),
        pred_classes=empty(0,dtype=int), pred_masks=empty(0,h,w))
    return Instances((h,w), **new) # OR: out.cat()


#######################################################
def Infer2Vid(src, data_dir, thd=0.6, rot=0):
    pred, meta = Setup4Infer(data_dir, thd)
    FCC = cv2.VideoWriter_fourcc(*'XVID')
    if os.path.isdir(src):
        imgs = sorted(glob(src+'/*.jpg'))
        sz = cv2.imread(imgs[0]).shape[1::-1] # (w,h)
        out = cv2.VideoWriter(src+'.avi', FCC, 1.0, sz)
        for i in tqdm(imgs, ascii=True):
            im = cv2.imread(i)
            out.write(Infer2Img(im, pred, meta, sz))
            if cv2.waitKey(2)==27: break
    elif os.path.isfile(src):
        cap = cv2.VideoCapture(src)
        assert not src.endswith('.avi')
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sz = (h,w) if rot else (w,h) # rotate 90 degree
        out = cv2.VideoWriter(src[:-4]+'.avi', FCC, 10.0, sz)
        for i in trange(N, ascii=True):
            res, im = cap.read(); assert res
            if rot: im = cv2.rotate(im, 0) # 90_CLOCKWISE
            out.write(Infer2Img(im, pred, meta, sz))
            if cv2.waitKey(2)==27: break
        cap.release()
    cv2.destroyAllWindows()


#######################################################
if __name__ == '__main__':
    data_dir = './coco_door'
    ym = './model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    Setup2Train(data_dir, ym, 30000)

    #src = 'VID_20200703_133959.mp4'
    #src = 'data_train/VID_20200703_163912'
    #Infer2Vid(src, data_dir, 0.6, rot=1)

