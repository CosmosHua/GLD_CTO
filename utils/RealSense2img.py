#!/usr/bin/python3
# coding: utf-8

import os, cv2
import time as tm
import numpy as np
import argparse, json
import pyrealsense2 as rs


##########################################################################################
# Create object for parsing command-line options
parser = argparse.ArgumentParser(description='Display depth stream in jet colormap.\
    Note: change resolution, format and fps to match the recorded stream!')
# Add argument which takes path to a bag file as an input
parser.add_argument('--fps', type=int, default=15, help='fps for record')
parser.add_argument('--res', type=tuple, default=(1280,720), help='resolution for record')
parser.add_argument('--blur', type=float, default=50, help='motion blur filter_threshold')
parser.add_argument('--save', action='store_true', help='extract and save images')
parser.add_argument('-s', '--source', default='0', help='device or bag file')
# Parse the command line arguments to an object
args = parser.parse_args(); #print(args)

device, res, fps = args.source, args.res, args.fps
device = False if device.endswith('.bag') and os.path.isfile(device) else True

config = rs.config() # Create a config object
if not device: # source: not from device
    # Config use a recorded file for playback
    rs.config.enable_device_from_file(config, args.source)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
else: # from realsense device
    config.enable_stream(rs.stream.depth, *res, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, *res, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.infrared, 1, *res, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, *res, rs.format.y8, fps)

pipeline = rs.pipeline() # Create pipeline
profile = pipeline.start(config) # Start streaming
depth_sensor = profile.get_device().first_depth_sensor()

# rs.align allows us to perform alignment to others frames.
align = rs.align(rs.stream.color) # align to color stream


##########################################################################################
def RS2img(save=args.save, ths=args.blur):
    #colorizer = rs.colorizer() # jet colormap
    JS = []; args.save, args.blur = save, ths; print(args)
    dst = str(int(tm.time())) if device else args.source[:-4]
    if save: os.makedirs(dst, exist_ok=True); os.chdir(dst)

    while True: # Streaming loop
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        if align: frames = align.process(frames)

        # Get a coherent pair of frames: depth and color
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color: continue

        # Convert images to numpy arrays
        #depth = colorizer.colorize(depth) # first colorize, then to numpy
        depth = np.asanyarray(depth.get_data()) # OR: first to numpy, then applyColorMap
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        #depth_color = cv2.cvtColor((depth/depth.max()*255).astype('uint8'), cv2.COLOR_GRAY2BGR)

        color = np.asanyarray(color.get_data())[...,::-1] # RGB->BGR
        #color = cv2.resize(color, depth.shape[1::-1])

        var = blur_score(color) # estimate motion blur
        depth_color = cv2.resize(depth_color, color.shape[1::-1])
        depth_color = cv2.addWeighted(depth_color, 0.6, color, 0.5, 0)
        #depth_color = np.vstack([depth_color, color])
        cv2.putText(depth_color, '%.2f'%var, (2,30), 4, 1, (222,)*3)
        cv2.imshow('depth_color', depth_color); key = cv2.waitKey(1)

        if device and not save:
            ir_left = frames.get_infrared_frame(1)
            ir_right = frames.get_infrared_frame(2)
            #Get_Intrinsics(ir_left, ir_right)
            ir_left = np.asanyarray(ir_left.get_data())
            ir_right = np.asanyarray(ir_right.get_data())
            IR = np.hstack((ir_left, ir_right))
            cv2.imshow('left+right', IR)
        if save and var>ths: # date = tm.ctime()[4:]
            t, blur = tm.time(), float('%.3f'%var); i = str(t)
            date = tm.strftime('%Y-%m-%d %H:%M:%S', tm.localtime(t))
            cv2.imwrite(i+'.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 96])
            cv2.imwrite(i+'.png', depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #color, depth = cv2.imread(i+'.jpg',-1), cv2.imread(i+'.png',-1)
            #print(color.dtype, color.shape, depth.dtype, depth.shape)
            JS.append(dict(color=i+'.jpg', blur=blur, date=date, depth=i+'.png'))
        if key==32: key = cv2.waitKey(0) # halt
        if key==27: cv2.destroyAllWindows(); break

    pipeline.stop() # Stop streaming
    with open(f'{dst}.json','w+') as ff:
        json.dump(JS, ff, indent=4); return JS


#######################################################
def Get_Intrinsics(left, right=None):
    prof_l = left.get_profile()
    prof_r = right.get_profile() if right else prof_l
    vs_prof_l = rs.video_stream_profile(prof_l)
    vs_prof_r = rs.video_stream_profile(prof_r)
    print(vs_prof_l.get_intrinsics()) # intrinsics
    print(vs_prof_r.get_intrinsics()) # intrinsics
    if right: print(prof_l.get_extrinsics_to(prof_r))


def blur_score(im, ksz=60): # estimate motion blur
    if type(im)==str and os.path.isfile(im):
        im = cv2.imread(im, 0) # gray
    elif im.ndim>2: # for BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if type(ksz)==int and ksz>0: # using FFT
        cy, cx = [i//2 for i in im.shape[:2]]
        fft = np.fft.fftshift(np.fft.fft2(im))
        fft[cy-ksz:cy+ksz, cx-ksz:cx+ksz] = 0
        fft = np.fft.ifft2(np.fft.ifftshift(fft))
        return np.mean(np.log(np.abs(fft)))*200
    else: # using Laplacian(ksize=1)
        return cv2.Laplacian(im, cv2.CV_64F).var()


##########################################################################################
if __name__ == '__main__':
    RS2img(save=True)

