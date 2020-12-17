#!/usr/bin/python3
# coding: utf-8

import os, json
import numpy as np
from copy import deepcopy


norm = np.linalg.norm; pre = 'New'
# self.list: [{id, class_id, position, direction, width}]
##########################################################################################
class objects:
    def __init__(self, bim):
        if type(bim)==str and os.path.isfile(bim):
            with open(bim, 'r+') as f: bim = json.load(f)
            self.list = bim if type(bim)==list else [bim]
        elif type(bim)==dict: self.list = deepcopy([bim])
        elif type(bim)==list: self.list = deepcopy(bim)
        else: self.list = [] # list of dicts
        for p in self.list: # pos/drt->np.array
            p['position'] = np.asarray(p['position'])
            p['direction'] = np.asarray(p['direction'])


    def __len__(self): return len(self.list)

    def __repr__(self): return str(self.list)

    def __iter__(self): yield from self.list

    def __getitem__(self, key): # list/np.array/dict
        if type(key)==str: return self.to_dict()[key]
        else: return self.list[key] # dict


    def find(self, item): # item: str
        if type(item)==dict: item = item['id']
        for dc in self.list:
            if dc['id']==item: return dc


    def to_dict(self, lst:list=None): # dict of lists
        lst = self.list if lst==None else lst; N = len(lst)
        idx = ['']*N; cid = np.zeros(N,int); wid = np.zeros(N)
        pos = np.zeros((N,3)); drt = np.zeros((N,3))
        for i,p in enumerate(lst):
            idx[i] = p['id']; cid[i] = p['class_id']; wid[i] = p['width']
            pos[i] = p['position']; drt[i] = p['direction'] #(N,3)
        return dict(id=idx, class_id=cid, position=pos, direction=drt, width=wid)


    def to_list(self, dct): # single dict->list
        dct = dct if type(dct)==dict else self.list[dct]
        idx, cid, wid = dct['id'], dct['class_id'], dct['width']
        pos, drt = dct['position'], dct['direction']
        return [idx, cid, pos, drt, wid] # dict->list


    def append(self, new, pre=pre):
        if len(self.list)>0:
            idx = self.list[-1]['id']
            idx = idx[len(pre):] if idx.startswith(pre) else '0'
            idx = pre + str(int(idx)+1)
        else: idx = pre + '1' # the first

        if type(new) in (list,tuple):
            cid, pos, drt, wid = new[:4]
            if type(wid)==np.ndarray: #(3,N)
                wid = np.cross(drt,[0,0,1]).T.dot(wid)
                wid = wid.max()-wid.min() # width-direction
            new = dict(id=idx, class_id=cid, position=pos,
                direction=drt, width=float(wid))
        if 'id' not in new.keys(): new['id'] = idx
        self.list += [new]; return len(self)-1


    # update prestored + append new objects
    def update(self, new, thd=(0.8,0.94), q=0.5):
        if len(self)<1: return self.append(new)
        if type(new)==dict:
            cid, pos = new['class_id'], new['position']
            drt, wid = new['direction'], new['width']
        else: cid, pos, drt, wid = new[:4] # list/tuple

        dis = norm(self['position'][:,:2]-pos[:2], axis=1)
        idx = dis.argsort()[:np.count_nonzero(dis<thd[0])]
        '''cos = self['direction'][idx,:2] # (N,2)
        cos = cos.dot(drt[:2])/norm(drt[:2])/norm(cos,axis=1)
        idx = idx[np.where(cos>thd[1])[0]] # direction'''
        if len(idx)<1: return self.append(new)

        for i in idx: # update prestored
            p = self.list[i]; dr = drt # for short
            if not p['id'].startswith(pre): return i
            dr = q*p['direction']+(1-q)*drt; dr /= norm(dr)
            wd = width(dr,wid) if type(wid)==np.ndarray else wid
            if wd>p['width']*0.5: #and p['class_id']==cid:
                p['width'] = q*p['width'] + (1-q)*float(wd)
                p['position'] = q*p['position'] + (1-q)*pos
                p['direction'] = dr; p['class_id'] = cid; return i


    # find by world coordinate, not update
    def update2(self, new, thd=(0.8,0.94)):
        assert len(self.list)>1
        if type(new)!=dict: pos, drt = new[1:3] # list
        else: pos,drt = new['position'], new['direction']
        
        dis = norm(self['position'][:,:2]-pos[:2], axis=1)
        idx = dis.argsort()[:np.count_nonzero(dis<thd[0])]
        '''cos = self['direction'][idx,:2] # (N,2)
        cos = cos.dot(drt[:2])/norm(drt[:2])/norm(cos,axis=1)
        idx = idx[np.where(cos>thd[1])[0]] # direction'''
        if len(idx)>0: return idx[0]


##########################################################################################
def width(drt, pts): # pts: (3,N)
    w = np.cross(drt,[0,0,1]).dot(pts)
    return w.max()-w.min()


def obj2pos(pos, obj):
    p = pos.position; q = pos.orientation
    p.x, p.y, p.z = obj['position']
    q.x, q.y, q.z = obj['direction']
    return pos # needless


FT = lambda x,d=4: tuple(round(i,d) for i in x)
##########################################################################################
def log_init(obj, log=None):
    log = log if type(log)==dict else {}
    for i in obj: # [cid,pos,drt;dp,iop]
        log[i['id']] = [dict(cid=i['class_id'],
        pos=FT(i['position']), drt=FT(i['direction']))]
    return log


def log_append(log, key, val, dp, iop=[]):
    if type(key)==dict: key = key['id']
    if val==None or type(dp)!=np.ndarray: return key
    val = dict(cid=val[0], pos=FT(val[1]), drt=FT(val[2]),
                dp=round(dp.mean(),4) )
    if len(iop)>0: val['iop'] = round(max(iop),4)
    if key in log: log[key].append(val)
    else: log[key] = [val] # add New
    return key, val # [cid,pos,drt;dp,iop]


def log_output(log, ff='seg'):
    ff = open(ff+'.log','w')
    for k,v in log.items():
        print('%s:'%k, file=ff)
        for i in v: print(i,file=ff)
    ff.close()

