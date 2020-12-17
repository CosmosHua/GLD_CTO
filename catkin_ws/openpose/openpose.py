import torch
import os, cv2
import numpy as np
from time import time 

from src import model
from src import util
from src.body import Body
from src.hand import Hand


##########################################################################################
class openpose(object):
    def __init__(self, root=None):
        if type(root)!=str:
            root = os.path.expanduser('~/openpose/model')
        self.body_estimation = Body(root+'/body_pose_model.pth')
        self.hand_estimation = Hand(root+'/hand_pose_model.pth')
        print(f'Torch device: {torch.cuda.get_device_name()}')


    def process_image(self, img, estimate_hand=False, count_time=True):
        try:
            t0 = time(); all_hand_peaks = []
            candidate, subset = self.body_estimation(img)
            if estimate_hand: # detect hand
                hands_list = util.handDetect(candidate, subset, img)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(img[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                    all_hand_peaks.append(peaks)
            if count_time:
                print('openpose: %.2f ms' % ((time()-t0)*1000))
            return candidate, subset, all_hand_peaks
        except:
            return None, None, None


    def draw_result(self, img, candidate , subset, all_hand_peaks ): 
        canvas = img.copy()
        canvas = util.draw_bodypose(canvas, candidate, subset) 
        if all_hand_peaks is not None and all_hand_peaks != []:
            canvas = util.draw_handpose(canvas, all_hand_peaks)
        return canvas
 
 
##########################################################################################
if __name__ == '__main__':
    wk = openpose()
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    while True:
        ret, img = cap.read()
        if not ret: continue
        img = img[:, 60:260].copy()
        candidate, subset, hands = wk.process_image(img, False, True)
        if candidate is None: continue
        canvas = wk.draw_result(img, candidate, subset, hands)
        canvas = cv2.resize(canvas, (800,960))
        cv2.imshow('demo', canvas)
        if cv2.waitKey(1)==27: break
    cap.release()
    cv2.destroyAllWindows()

