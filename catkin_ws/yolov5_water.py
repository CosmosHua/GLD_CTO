#!/usr/bin/python3
# coding: utf-8

import matplotlib, gi
matplotlib.use('Agg') #'TkAgg'
#gi.require_version('Gtk','2.0')

import time as tm
import numpy as np
import os, cv2, sys
import socket, json, math

sys.path.append('yolov5')
#from yolov5.infer import yolov5_det
from yolov5.infer1 import yolov5_det
from rs_cam import rs_init, rs_rgbd
from speech_baidu import *


##########################################################################################
def Quart2Euler(qt):
    if type(qt) in (list,tuple): x,y,z,w = qt
    elif type(qt)==dict:
        x,y,z,w = qt['x'],qt['y'],qt['z'],qt['w']
    yaw = math.atan2(2*(w*z+x*y), 1-2*(z*z+y*y))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(2*(w*y-x*z))
    return yaw, roll, pitch # radian: [-pi,pi]


def Euler2Quart(yaw, roll, pitch):
    euler = [yaw/2, roll/2, pitch/2]
    siny, sinr, sinp = np.sin(euler)
    cosy, cosr, cosp = np.cos(euler)
    x = sinp*siny*cosr + cosp*cosy*sinr
    y = sinp*cosy*cosr + cosp*siny*sinr
    z = cosp*siny*cosr - sinp*cosy*sinr
    w = cosp*cosy*cosr - sinp*siny*sinr
    return x, y, z, w # euler: radian


##########################################################################################
def tcp_init(addr=None): # initialize tcp
    #ip = input('IP:'); port = int(input('Port:'))
    if type(addr)!=tuple: addr = ('192.168.10.10', 31001)
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.connect(addr); return tcp # tcp.settimeout(0.2)


# bytes.decode(encoding='utf-8')->str
def tcp_get(tcp=None, show=False, bt=0.2):
    if type(tcp)!=socket.socket: tcp = tcp_init()
    t = tcp.gettimeout(); tcp.settimeout(bt)
    try: # for case of tcp.recv() time_out
        res = tcp.recv(4096) # recv() is block func
        while res[-1:]!=b'\n': res += tcp.recv(4096)
    except: tcp.settimeout(t); return [{}] # time_out
    res = b','.join(res.split(b'\n')[:-1]) #.decode()
    #res = ','.join(res.decode().split('\n')[:-1])
    res = json.loads(b'[%s]'%res) # f'[{res}]'
    if show: # print(tm.ctime())
        for i in res: print(json.dumps(i,indent=4))
    tcp.settimeout(t); return res # list of dicts


# str.encode(encoding='utf-8')->bytes
def tcp_cmd(tcp=None, cmd=None, show=False, buf=1):
    if type(tcp)!=socket.socket: tcp = tcp_init()
    if buf: tcp_get(tcp, False, 0.2) # flush
    if type(cmd)!=str: cmd = input('[cmd]: ')
    print('=>', cmd.split('&')[:2])
    tcp.send(f'/api/{cmd}'.encode())
    return tcp_get(tcp, show, 1)[0] # dict


##########################################################################################
def format_pos(ps): # convert format
    if 'theta' not in ps: # need: 'orientation'
        ps.update(ps['position']) # extract 'position'
        ps['theta'] = Quart2Euler(ps['orientation'])[0]
        ps.pop('position'); ps.pop('orientation')
    return ps # keys: ('x','y','z','theta')


def rename_marker(tcp=None, ren={}):
    mark = tcp_cmd(tcp,'markers/query_list')['results']
    for k,v in ren.items(): # key=old, val=new
        if k not in mark: continue
        p = format_pos(mark[k]['pose'])
        f = f"?floor={mark[k]['floor']}"
        n = f"&name={v}&type={mark[k]['key']}"
        p = f"&x={p['x']}&y={p['y']}&theta={p['theta']}"
        tcp_cmd(tcp, f'markers/insert_by_pose'+f+n+p)
        tcp_cmd(tcp, f'markers/delete?name={k}')


########################################################
# valid: only all passages are rectangular
def pos_cost(a, b, axis=None): # a=current, b=target
    a, b = format_pos(a), format_pos(b)
    x, y = b['x']-a['x'], b['y']-a['y']
    t = np.arctan(y/x)-(axis if axis==None else a['theta'])
    cos = abs(np.cos(t)); sin = abs(np.sin(t))
    dis = np.linalg.norm([x,y]); return dis*(sin+cos)


# suffix: W=Window, I=Indoor, G=Glass_wall
# Eg. TG={'522':{'I':['522I'],'W':['522W']}}
def arrange_target(tcp=None): # NOT optimal
    # OR: 'map/get_current_map'->'results'->'floor'
    st = tcp_cmd(tcp,'robot_status')['results']
    cf = str(st['current_floor']); cp = st['current_pose']
    mark = tcp_cmd(tcp,'markers/query_list')['results']
    axis = format_pos(mark['home']['pose'])['theta']

    door = [k for k in mark if k[0]==cf and k.isdigit()]
    door = {k:mark[k]['pose'] for k in door}; TG = {}
    for i in range(len(door)):
        m = {pos_cost(cp,v,axis):k for k,v in door.items()}
        m = m[min(m)]; cp = door[m]; door.pop(m); TG[m] = {}
        TG[m]['I'] = [k for k in mark if m+'I' in k] # indoor
        TG[m]['W'] = [k for k in mark if m+'W' in k] # window
        TG[m]['G'] = [k for k in mark if m+'G' in k] # glass_wall
    print(f'F{cf}:', '->'.join(TG)); return cf, TG


SF = '&distance_tolerance=0.05&theta_tolerance=0.01'
MV = lambda m,a: f'move?marker={m}&angle_offset={a}'+SF
##########################################################################################
def abort(k, tcp=None):
    if k==27: # emergency->abort
        tcp_cmd(tcp, 'estop?flag=true')
        cv2.destroyAllWindows(); speak_('是否中止巡检')
        k = 27 if 'y' in input('Abort?[y/n]: ') else 1
        if k==27: tcp_cmd(tcp, 'move/cancel')
        tcp_cmd(tcp, 'estop?flag=false')
    elif k==0: tcp_cmd(tcp, 'move/cancel')
    return k # 0=fail, 27=abort, 1=continue


# return: 0=fail, 27=abort, 1; str=info
def inspect(det, tcp, cmd, PA, js={}, mk=''):
    if type(cmd)==str: # pre-action
        st = tcp_cmd(tcp, cmd); af = -np.pi/2 # move
        if not mk and not st['error_message']: # valid cmd
            res = [i for i in cmd.split('&') if 'marker=' in i]
            if res: mk = res[0].split('=')[1] # get marker
        ofs = [i for i in cmd.split('&') if 'angle_offset' in i]
        if ofs: ofs = abs(float(ofs[0].split('=')[1])-af)<0.1
    first = mk.isdigit() and type(cmd)==str and ofs # ofs=1,0,[]
    #if first: speak('即将前往%s会议室'%' '.join(list(mk)))
    while True: # notification maybe invalid!
        k = det_show(PA) # just show, no det
        if k==27 and abort(k,tcp)==27: return k

        st = tcp_get(tcp)[-5:]; st.reverse() # latest
        if len(st[0])<1: continue # fps=min(1/bt,PA.fps)
        nt = [i for i in st if i['type']=='notification']
        st = [i for i in st if i['type']=='callback']
        nt = nt[0] if nt else {'code':'', 'description':''}
        st = st[0]['results'] if st else {'move_status':''}
        #if st['move_status']: print('\tstatus:', st['move_status'])
        if nt['code']: print('\t%s:'%nt['code'], nt['description'])
        if nt['code']=='01203' and 'I' in mk: speak_('小小爱无法进门')
        if nt['code']=='01200': speak('哎呀，被挡住啦，请让一下')
        if nt['code']=='01002' and first: # for 1st arrival
            speak_('小小爱已顺利到达'+' '.join(list(mk)))

        if st['move_status']=='failed' or nt['code'] in \
           ('01003','01203'): return abort(0,tcp) #'01005'
        if st['move_status']=='succeeded': # det_infer
            if not (mk and js): return st['current_pose']
            return abort(det_show(PA, det, js, mk), tcp)


# return: 0=fail, 27=abort, 1, str; None
def inspects(det, tcp, PA, mk, js, A=None):
    assert type(js)==dict; af = -np.pi/2; d = 0.2
    assert type(PA) in (tuple,list) and len(PA)==2
    # only wood_door has indoor and window views
    if mk+'I' in MARK or 'G' in mk: # only once
        k = inspect(det, tcp, MV(mk,af), PA, js, mk)
        if k in (0,27): return k # 0=fail, 27=abort
    elif 'W' in mk: # window: rotate around camera
        k = inspect(det, tcp, MV(mk,af), PA, js, mk)
        if k in (0,27) or LB[1] in js.values(): return k
        pos = format_pos(MARK[mk]['pose']); t = pos['theta']
        cx, cy = pos['x']+np.cos(t)*d, pos['y']+np.sin(t)*d
        A = [-45,45] if A is None else sorted(A) # angles
        for a in np.asarray(A)*np.pi/180: # scan
            th, a = (t+a)+np.pi, (t+a)+af # reverse
            x, y = cx+np.cos(th)*d, cy+np.sin(th)*d
            cmd = 'move?location=%.3f,%.3f,%.7f'%(x,y,a)
            k = inspect(det, tcp, cmd+SF, PA, js, mk)
            if k in (0,27) or LB[1] in js.values(): return k
    else: # indoor/glass: rotate around robot center
        A = [0,-30,30] if A is None else [0]+sorted(A)
        for a in np.asarray(A)*np.pi/180 + af:
            k = inspect(det, tcp, MV(mk,a), PA, js, mk)
            if k in (0,27) or LB[1] in js.values(): return k


MARK = {} #tcp_cmd(tcp,'markers/query_list')['results']
Date = lambda: tm.strftime('%Y-%m-%d-%H:%M:%S', tm.localtime())
##########################################################################################
def det_infer(det, im, dst=None):
    if type(im)==str: im = cv2.imread(im)
    im, res, dt = det.infer1(im, True, True, False)
    print('\t%s: %.2fms'%(dst, dt), res) # for infer1
    if type(dst)!=str: return im, res, dt # res=dict

    tc = (0,255,255) if len(res)>0 else (255,)*3
    ret = LB[len(res)>0]; info = dst[:-4]+': '+ret
    cv2.rectangle(im, (0,2), (len(info)*16,35), (88,)*3, -1)
    cv2.putText(im, info, (2,26), 4, 0.8, tc) # ret=str
    cv2.imwrite(DST+'/'+dst, im); return im, ret, res, dt


def det_show(PA, det='', js={}, mk=''): # js=dict
    im, dst = rs_rgbd(*PA,0), f'{Date()} {mk}.jpg'
    if det and js and dst: # det or not
        im, js[dst] = det_infer(det, im, dst)[:2]
    cv2.imshow('det',im); return cv2.waitKey(10)


LB = {1:'in use', 0:'vacant maybe', -1:'-'}; DST = '.'
LK = lambda x: 1 if LB[1] in x else 0 if LB[0] in x else -1
VOC = {1:'检测到有人在使用%s', 0:'可能无人使用%s', -1:'尚未检查%s'}
##########################################################################################
def water_cruise(det):
    tcp = tcp_init(); PA = rs_init()[1:] # init
    det_infer(det, rs_rgbd(*PA, mod=0)) # tryout

    cf, target = arrange_target(tcp); global DST
    DST = Date()[:10]+f'_F{cf}'; tt = Date()[:-3]
    os.makedirs(DST, exist_ok=True); #os.chdir(DST)

    out = f'{DST}/{DST}.json'; JS = {tt:{}} # reverse
    if os.path.isfile(out): # load history->reserve
        with open(out,'r+') as f: JS.update(json.load(f))

    speak(f'开始巡检第{cf}层的会议室'); global MARK
    MARK = tcp_cmd(tcp,'markers/query_list')['results']
    tcp_cmd(tcp,'estop?flag=false') # disable estop
    tcp_cmd(tcp,'request_data?topic=robot_status') # 2Hz
    for ot, giw in target.items(): # giw=dict(G,I,W)
        speak('即将前往%s会议室'%' '.join(list(ot)))
        js = JS[tt][ot] = {'result': LB[-1]} # init
        k = inspects(det, tcp, PA, ot, js, A) # outdoor
        for x in giw['G']: # loop: glass_walls
            if LB[1] in js.values() or k==27: break
            k = inspects(det, tcp, PA, x, js, A) # js/JS
        for x in giw['I']: # loop: indoors, fail->skip
            if LB[1] in js.values() or k in (0,27): break
            k = inspects(det, tcp, PA, x, js, A) # js/JS
        for x in giw['W']: # loop: windows
            if LB[1] in js.values() or k==27: break
            k = inspects(det, tcp, PA, x, js, A) # js/JS
        js['result'] = LB[LK(js.values())] # update JS
        speak(VOC[LK(js.values())]%' '.join(list(ot)))
        if k==27: break # emergency->abort->estop
    tcp_cmd(tcp,'estop?flag=true') # enable estop
    Speak_([f'第{cf}层的会议室巡检结束','是否回到充电桩'])

    with open(out,'w+') as f: json.dump(JS,f,indent=4)
    cv2.destroyAllWindows() #os.chdir('..') # for input
    if input('Task done! Back home?[y/n]: ')!='n':
        tcp_cmd(tcp,'estop?flag=false') # disable estop
        inspect(det, tcp, 'move?marker=home', PA)
    tcp_cmd(tcp,'estop?flag=true'); tcp.close(); PA[0].stop()


##########################################################################################
if __name__ == '__main__':
    #while True: tcp_cmd(show=True)
    model = 'yolov5/yolov5x.pt'; k=7
    det = yolov5_det(model, cls=[0])
    while k!='n': # 0=person, 63=laptop
        water_cruise(det)
        k = input('\nAgain?[y/n]: ')

