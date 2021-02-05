#!/usr/bin/python3
# coding: utf-8

import cv2, sys
import numpy as np
import pyrealsense2 as rs


##########################################################################################
def rs_init(res=(1280,720), fps=15):
    config, pipeline = rs.config(), rs.pipeline() # Create
    config.enable_stream(rs.stream.depth, *res, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, *res, rs.format.bgr8, fps)
    #config.enable_stream(rs.stream.color, *res, rs.format.rgb8, fps)
    #config.enable_stream(rs.stream.infrared, 1, *res, rs.format.y8, fps)
    #config.enable_stream(rs.stream.infrared, 2, *res, rs.format.y8, fps)
    profile = pipeline.start(config) # Start streaming
    depth_sensor = profile.get_device().first_depth_sensor()
    align = rs.align(rs.stream.color) # align to color stream
    return config, pipeline, align # pipeline.stop()


def rs_rgbd(pipeline, align, mod=0):
    assert type(pipeline)==rs.pipeline
    frames = pipeline.wait_for_frames()
    if align: assert type(align)==rs.align
    if align: frames = align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color and not depth: return
    if mod in (0,2): color = np.asanyarray(color.get_data())
    if mod in (1,2): depth = np.asanyarray(depth.get_data())
    return color if mod==0 else depth if mod==1 else (color,depth)


########################################################
def rs_yolov5(model, cls=None):
    sys.path.append('yolov5')
    #from yolov5.infer import yolov5_det
    from yolov5.infer1 import yolov5_det

    det = yolov5_det(model, cls=cls) # init yolov5
    #Infer = lambda x: det.infer(x, '.', False)[0]
    Infer1 = lambda x: det.infer1(x, True, True, False)
    config, pipeline, align = rs_init(); k = 0
    while k!=27:
        im = rs_rgbd(pipeline, align, mod=0)
        #cv2.imwrite(dst,im); res = Infer(dst)
        im, res, dt = Infer1(im); #cv2.imwrite(dst,im)
        if res: print('%.2fms:'%dt, res) # for Infer1
        cv2.imshow('det', im); k = cv2.waitKey(5)
    cv2.destroyAllWindows(); pipeline.stop()


# pip3 install facenet-pytorch
########################################################
def rs_face(show=1):
    from facenet import MTCNN
    from facenet import face_det

    mtcnn = MTCNN(); k = 0
    pipeline, align = rs_init()[1:]
    while k!=27:
        im = rs_rgbd(pipeline, align, mod=0)
        im = face_det(im, mtcnn, show=show)[0]
        cv2.imshow('det',im); k = cv2.waitKey(5)
    cv2.destroyAllWindows(); pipeline.stop()


##########################################################################################
if __name__ == '__main__':
    #model = 'yolov5/yolov5x.pt'
    #rs_yolov5(model, cls=[0,63])
    rs_face(3)

