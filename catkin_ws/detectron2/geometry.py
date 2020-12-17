#!/usr/bin/python3
# coding: utf-8

import cv2
import numpy as np
import scipy.linalg as la


CK = [263.6988, 0, 344.7538, 0, 263.6988, 188.0399, 0, 0, 1]
########################################################
# 2D axes: origin lies top-left
#   u-axis: horizontal->rightward
#   v-axis: vertical->downward
# 3D axes: origin lies top-left
#   x-axis: horizontal->rightward
#   y-axis: vertical->downward
#   z-axis: horizontal->forward
def Camera_2Dto3D(u, v, dp):
    u,v = [np.asarray(i,int) for i in (u,v)]
    dp = np.asarray(dp); shp = dp.shape
    fx,_,cx,_,fy,cy = CK[:6] # cam intrinsics
    z = dp[v,u] if len(shp)>1 else dp # meter
    z = z * (1E-3 if z.dtype==np.uint16 else 1)
    x = z * (u-cx)/fx; y = z * (v-cy)/fy
    return np.asarray([x,y,z]) #(3,N)


def Camera_3Dto2D(pt):
    fx,_,cx,_,fy,cy = CK[:6] # cam intrinsics
    assert type(pt)==np.ndarray; x,y,z = pt[:3] #(3,N)
    zz = np.where(z!=0, z, np.finfo(float).eps)
    u = (x*fx + z*cx)/zz; v = (y*fy + z*cy)/zz
    return np.asarray([u,v]) #(2,N)


########################################################
# Camera extrinsics/extrinsic parameters
# quat = np.asarray([w,x,y,z]), t = [x,y,z]
def Robot2World_TRMatrix(quat, t=0):
    q = np.asarray(quat); n = np.dot(q,q) # Quaternion
    t, dim = ([0]*3,3) if type(t) in (int,float) else (t,4)
    if n<np.finfo(q.dtype).eps: return np.identity(dim)
    q *= np.sqrt(2.0/n); q = np.outer(q,q)
    TR = np.asarray( # transform/rotation matrix
        [[1.0-q[2,2]-q[3,3], q[1,2]+q[3,0], q[1,3]-q[2,0], 0.0],
         [q[1,2]-q[3,0], 1.0-q[1,1]-q[3,3], q[2,3]+q[1,0], 0.0],
         [q[1,3]+q[2,0], q[2,3]-q[1,0], 1.0-q[1,1]-q[2,2], 0.0],
         [t[0], t[1], t[2], 1.0]], dtype=q.dtype).T
    return TR if dim>3 else TR[:3,:3] # extrinsics


def Camera2World(TR, pt, ofs=0.35): #(3,N)
    #x,y,z = pt[:3]; x,y,z = z,-x,ofs-y # Cam->Robot
    #pt = np.asarray([x,y,z]).reshape([3,-1]) # (3,N)
    pt = np.asarray([pt[2], -pt[0], ofs-pt[1]])
    pt = np.insert(pt[:3], 3, values=1.0, axis=0)
    return TR.dot(pt[:4])[:3] # Robot->World


def World2Camera(TR, pt, ofs=0.35): #(3,N)
    #x,y,z = pt[:3]; x,y,z = -y,ofs-z,x # Robot->Cam
    pt = np.insert(pt[:3], 3, values=1.0, axis=0)
    pt = np.linalg.pinv(TR).dot(pt)[:3] # Robot
    return np.asarray([-pt[1], ofs-pt[2], pt[0]])


########################################################
def PCA(pt, rank=0):
    dim, N = pt.shape #(3,N) -> normalized
    pt = pt - pt.mean(axis=1, keepdims=True)
    cov = pt.dot(pt.T) / (N-1) # covariance matrix
    val, vec = np.linalg.eig(cov) # np.linalg.svd
    rank = dim if rank<1 else rank # dim>=max(rand)
    idx = val.argsort()[::-1][:rank] # descending
    return val[idx], vec[:,idx].T #(rank,dim)


def LeastSq(pt, y=0, T=True): #(3,N): ax+by+cz+d=0
    if T: x = np.insert(pt.T, pt.shape[0], 1, axis=1)
    else: x = np.insert(pt, pt.shape[1], 1, axis=1)
    if type(y)!=np.ndarray:y = np.ones(x.shape[0],1)*y
    sol, r, rank, s = la.lstsq(x, y) # x:(N,4)
    return sol # (a,b,c,d), x.dot(sol)=>y


# uv: (2,N), u=uv[0], v=uv[1]
# TR: camera extrinsic parameters
# plane: ax+by+cz+d=0, n=(a,b,c), d=-n*x0
def uvRay2Plane(uv, TR, plane):
    P1 = Camera_2Dto3D(0,0, 0.0) # camera
    P1 = Camera2World(TR, P1) # world, (3,)
    P2 = Camera_2Dto3D(*uv, 1.0) # camera
    P2 = Camera2World(TR, P2) # world, (3,)
    #P1,P2 = [np.asarray(i) for i in (P1,P2)]
    n, d = np.asarray(plane[:3]), plane[3:]
    d = -n.dot(d[:3]) if len(d)>1 else d
    k = -(n.dot(P1)+d)/n.dot(P2-P1) # ratio
    return P1 + k*(P2-P1)


########################################################
if __name__ == '__main__':
    u = range(4); v = range(1,5); d = range(-1,3)
    pc = Camera_2Dto3D(u,v,d); print('cam3d:\n', pc)
    uv = Camera_3Dto2D(pc); print('cam2d:\n', uv)
    # uv=[-0. 0. 2. 3.], [ 1. 0. 3. 4.]

    q = np.asarray([np.pi/4, 0, 0, np.pi/4]) # z-axis 90
    t = np.asarray([1,2,3]); pc = np.asarray([1,1,1])
    T = Robot2World_TRMatrix(q,t); print('TR:\n', T)

    pt = Camera2World(T,pc); print('world:\n', pt)
    pc = World2Camera(T,pt); print('cam3d:\n', pc)
    # pt=[2. 3. 2.85], pc=[1. 1. 1.]

    uv = Camera_3Dto2D(pc); print('cam2d:\n', uv)
    p1 = Camera2World(T,Camera_2Dto3D(0,0,0)); print(p1,pt)
    pt = uvRay2Plane(uv, T, [0,0,1,0]); print('world:\n', pt)
    # uv=[697.7875 540.5965], pt=[2. 3. 2.85]
    # p1=[1. 2. 3.85], uv_ray_pt=[4.85 5.85 0.]

    v,vc = PCA(np.random.rand(3,100)); print(v,'\n',vc)
