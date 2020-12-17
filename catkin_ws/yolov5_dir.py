#!/usr/bin/python3
# coding: utf-8

import os, sys, json
from glob import glob
from time import sleep
sys.path.append('yolov5')
#from yolov5.infer import yolov5_det
from yolov5.infer1 import yolov5_det


##########################################################################################
def det_dir(src, model, cls=None, show=True, gap=1):
    try:
        det = yolov5_det(model, cls=cls); results = {}
        infer = lambda x: det.infer(x,src,show,False)[0]
        infer1 = lambda x: det.infer1(x,True,True,show)
        while True:
            sleep(gap)
            ff = sorted(glob(src+'/*.jpg'))
            for i in ff:
                if i.endswith('_det.jpg') or i[:-4]+'_det.jpg' in ff: continue
                #res = 'Using' if 'person' in infer(i) else 'Unused maybe'
                res = 'Using' if len(infer(i))>0 else 'Unused maybe'
                key = os.path.basename(i).split(' ')[0]
                if key not in results or res=='Using':
                    results[key] = res
    finally:
        rec = tm.strftime("%Y-%m-%d %H:%M:%S", tm.localtime())
        with open(src+f'/{rec}.json','w+') as js:
            json.dump(results, js, indent=4)
 

##########################################################################################
if __name__ == '__main__':
    import sys; arg = sys.argv; root = 'yolov5/'
    src = arg[1] if len(arg)>1 else root+'../test'
    model = arg[2] if len(arg)>2 else root+'yolov5x.pt'
    det_dir(src, model, [0,63]) # 0=person, 63=laptop

