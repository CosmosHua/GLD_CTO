#!/usr/bin/python3
# coding: utf-8

import os, cv2, json
from pathlib import Path
from random import randint
from time import time

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, plot_one_box, strip_optimizer, set_logging, increment_dir
from utils.torch_utils import select_device, load_classifier, time_synchronized


##########################################################################################
@torch.no_grad()
class yolov5_det(object):
    def __init__(self, wt, conf=0.25, iou=0.45, cls=None, size=640, augment=False, agnostic_nms=False):
        if type(wt)!=str or not os.path.isfile(wt):
            for wt in ['yolov5x.pt', 'yolov5l.pt', 'yolov5m.pt', 'yolov5s.pt']:
                if os.path.isfile(wt): break
        self.weight = wt; assert os.path.isfile(wt)
        print(f'YOLOv5: {torch.cuda.get_device_name()}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type!='cpu' # only CUDA supports half-precision
        self.model = model = attempt_load(wt, map_location=self.device) # FP32
        if self.half: self.model.half() # convert to FP16

        self.names = model.module.names if hasattr(model,'module') else model.names
        self.colors = [[randint(0,255) for _ in range(3)] for _ in self.names]

        self.iou_thres = iou; self.conf_thres = conf
        self.classes = cls; self.agnostic_nms = agnostic_nms
        self.augment = augment; s = int(model.stride.max())
        self.imsz = int(size if size%s==0 else (size//s)*s)

        #img = torch.zeros((1, 3, self.imsz, self.imsz), device=self.device)
        #model(img.half() if self.half else img # run once for test


    def infer(self, src, dst=None, show='det', over=True):
        src = str(src); save = False
        if type(dst)==str:
            os.makedirs(dst, exist_ok=True); dst = Path(dst); save = True
        webcam = src.isdigit() or src.startswith(('rtsp://','http://'))

        if webcam: # Set Dataloader
            cudnn.benchmark = True # speedup for constant img_size
            data = LoadStreams(src, img_size=self.imsz)
        else: data = LoadImages(src, img_size=self.imsz)

        # Run inference
        RST = []; t0 = time()
        vid_path, vid_writer = None, None
        names = self.names; colors = self.colors
        for path, img, imgs, vid_cap in data:
            img = torch.from_numpy(img).to(self.device)
            # convert uint8 to fp16/fp32, [0,255] to [0,1.0]
            img = (img.half() if self.half else img.float())/255.0
            if img.ndimension()==3: img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]

            # Apply NMS. pred: list of N tensor, N=batch_size
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                            classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Process detections per batch of images
            for i, det in enumerate(pred): # webcam: batch_size>=1 for multi-stream
                p, im = (Path(path[i]+'.mp4'), imgs[i].copy()) if webcam else (Path(path), imgs)

                res = {} # img.shape[2:],img.dtype
                if det is not None and len(det):
                    # Rescale boxes from img_shape to im_shape(original)
                    det[:,:4] = scale_coords(img.shape[2:], det[:,:4], im.shape).round()
                    for *xyxy, prob, c in reversed(det): # det[0]:(x1,y1,x2,y2,prob,cls)
                        if save or show: # Add bbox to image
                            c = int(c); label = '%s %.2f' % (names[c], prob)
                            plot_one_box(xyxy, im, label=label, color=colors[c], line_thickness=2)
                    res = {names[int(c)]:int((det[:,-1]==c).sum()) for c in det[:,-1].unique()}
                RST.append(res); print('%.2fms:'%((t2-t1)*1000), res)

                if show: # show results
                    cv2.imshow(str(show), im)
                    if cv2.waitKey(1)==27: return

                if save: # save results
                    save_path = str(dst/p.name)
                    if data.mode=='images' and not webcam:
                        if not over and os.path.isfile(save_path):
                            save_path = save_path[:-4]+'_det.jpg'
                        cv2.imwrite(save_path, im)
                    else:
                        if vid_path != save_path: # init new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release() # release previous video writer
                            fourcc = 'mp4v' # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w,h))
                        vid_writer.write(im)
        return RST


##########################################################################################
if __name__ == '__main__':
    import sys; arg = sys.argv
    src = arg[1] if len(arg)>1 else '../Test.mp4'
    dst = arg[2] if len(arg)>2 else '../test'
    det = yolov5_det('yolov5x.pt', cls=[0,63])
    det.infer(src, dst) # 0=person, 63=laptop

