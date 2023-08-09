import numpy as np
import open3d as o3d
import copy
import math
import time
import cv2
from plyfile import *
import os

from tqdm import trange
import matplotlib.pyplot as plt
from functools import reduce
from copy import deepcopy
import pandas as pd
import trimesh
# import operator

from polygonal_approximation import thick_polygonal_approximate

def dcmp(x):
    if abs(x) < 1e-6:
        return 0
    else:
        return -1 if x<0 else 1

def onSegment(p1, p2, q):
    t1 = p1-q
    t2 = p2-q
    crossProduct = t1[0]*t2[1] - t1[1]*t2[0]
    product = t1[0]*t2[0] + t1[1]*t2[1]
    return dcmp(crossProduct) == 0 and dcmp(product) <= 0

def inPolygon(polygon, p):
    Flag = np.zeros([p.shape[0]])
    for k in trange(p.shape[0]):
        flag = False
        lines = zip(range(len(polygon)), range(1, len(polygon)+1))
        for i, j in lines:
            if j == len(polygon):
                j = 0
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[j])
            if onSegment(p1, p2, p[k,:]):
                Flag[k] = True
                break
            # if (dcmp(p1[1]-p[k,1])>0 != dcmp(p2[1]-p[k,1])>0) and (dcmp(p[k,0] - (p[k,1]-p1[1])*(p1[0]-p2[0])/(p1[1]-p2[1])-p1[0])<0):
            if ((p1[1]>p[k,1]) != (p2[1]>p[k,1])) and (p[k,0] < (p[k,1]-p1[1])*(p1[0]-p2[0])/(p1[1]-p2[1])+p1[0]):
                flag = not flag
        Flag[k] = flag
    return Flag

def threshold(img, th):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])[1:]
    v1=0
    v2=255
    p1=0
    p2=0
    for i in range(th):
        if hist[i] > p1:
            v1 = i
            p1 = hist[i]
    for j in range(254, th, -1):
        if hist[j] > p2:
            v2 = j
            p2 = hist[j]
    pth = int((v1*v2)**0.5)
    return pth

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def load_pcd(filename, tdict):
    print('loading pcd')
    st=time.time()

    pcdo=o3d.geometry.TriangleMesh()

    plydata = PlyData.read(filename)
    if 'face' in plydata:
        if len(plydata['face']['vertex_indices'])>0:
            pcdo=o3d.geometry.TriangleMesh()
            plytype='trianglemesh'
            mesh=plydata['face']['vertex_indices']
            pcdo.triangles=o3d.utility.Vector3iVector(mesh)
        else:
            pcdo=o3d.geometry.PointCloud()
            plytype='pointcloud'
    else:
        pcdo=o3d.geometry.PointCloud()
        plytype='pointcloud'
    xlist = plydata['vertex']['x']
    ylist = plydata['vertex']['y']
    zlist = plydata['vertex']['z']
    xyzo=np.array([xlist,ylist,zlist])
    xyzo=xyzo.transpose()
    mean=np.mean(xyzo,axis=0)
    xyzo=xyzo-mean

    if plytype=='pointcloud':
        rlist = plydata['vertex']['red']
        glist = plydata['vertex']['green']
        blist = plydata['vertex']['blue']
        rgbo=np.array([rlist,glist,blist])
        rgbo=rgbo.transpose()/255
        pcdo.colors=o3d.utility.Vector3dVector(rgbo)
        pcdo.points=o3d.utility.Vector3dVector(xyzo)
    else:
        pcdo.vertices=o3d.utility.Vector3dVector(xyzo)
        
    et=time.time()
    tdict['loading pcd']=et-st

    return pcdo, plytype

def rigid_transformation(pcdo, tdict):
    print('rigid transformation')
    st=time.time()

    xyzo = np.asarray(pcdo.points)

    #平面拟合,旋转
    X=copy.deepcopy(xyzo)
    X[:,2]=1
    Z=xyzo[:,2]
            
    A=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Z)
    a=A[0]
    b=A[1]

    axis=np.array([[b/(a**2+b**2)**0.5],[-a/(a**2+b**2)**0.5],[0]])
    angle=-1*math.acos(1/(a**2+b**2+1)**0.5)
    aa=np.dot(axis,angle)

    pcd1=copy.deepcopy(pcdo)
    R=pcd1.get_rotation_matrix_from_axis_angle(aa)
    pcd1.rotate(R,center=(0,0,0))

    et=time.time()
    tdict['rigid ransforming']=et-st

    return pcd1

def projection(pcd1, plytype, margin, tdict):
    print('projection')
    st=time.time()

    if plytype=='trianglemesh':
        resolution=[300,400]
        xyz1=np.asarray(pcd1.vertices)
    else:
        resolution=[300,400,3]
        print(pcd1)
        xyz1=np.asarray(pcd1.points)
        rgb1=np.asarray(pcd1.colors)
    minx=np.min(xyz1[:,0])
    maxx=np.max(xyz1[:,0])
    miny=np.min(xyz1[:,1])
    maxy=np.max(xyz1[:,1])
    mm = np.asarray([[minx, maxx], [miny, maxy]])
    dx=maxx-minx
    dy=maxy-miny
    if dx/dy>=(resolution[1]-2*margin)/(resolution[0]-2*margin):
        dp=dx/(resolution[1]-2*margin)
    else:
        dp=dy/(resolution[0]-2*margin)
    projectionimg=np.zeros(resolution)
    projectionimg=projectionimg.astype(np.uint8)

    et=time.time()
    tdict['projection']=et-st

    return projectionimg, mm, dp

def edge_extraction(path, pcd1, plytype, projectionimg, mm, dp, margin,  
tdict):
    print('extract edges')
    st=time.time()

    minx = mm[0, 0]
    maxy = mm[1, 1]
    resolution = projectionimg.shape

    if plytype=='trianglemesh':
        xyz1=np.asarray(pcd1.vertices)
        edge_manifold_boundary = pcd1.is_edge_manifold(allow_boundary_edges=False)
        if not edge_manifold_boundary:
            edges = pcd1.get_non_manifold_edges(allow_boundary_edges=False)
            
            edges_arr=np.asarray(edges)
            coords_arr=[]
            for i in range(len(edges_arr)):
                e=edges_arr[i]
                pixelx1=int((xyz1[e[0]][0]-minx)/dp)-1+margin
                pixely1=int((maxy-xyz1[e[0]][1])/dp)-1+margin
                pixelx2=int((xyz1[e[1]][0]-minx)/dp)-1+margin
                pixely2=int((maxy-xyz1[e[1]][1])/dp)-1+margin
                coords=[pixelx1,pixely1,pixelx2,pixely2]
                coords_arr.append(coords)
            coords_arr=np.asarray(coords_arr)
            for x1, y1, x2, y2 in coords_arr:
                cv2.line(projectionimg, (x1, y1), (x2, y2), 255, 2)
                cv2.imwrite(path+'/temp.png',projectionimg)
                
    elif plytype=='pointcloud':
        xyz1=np.asarray(pcd1.points)
        rgbo=np.asarray(pcd1.colors)
        
        label_list = np.zeros(projectionimg.shape[:2])
        for i in range(len(xyz1)):
            lx = int((xyz1[i][0]-minx)/dp)-1
            ly = int((maxy-xyz1[i][1])/dp)-1
            if not label_list[lx, ly]:
                projectionimg[ly+margin][lx+margin] = rgbo[i]*255
                label_list[lx][ly] = 1
        
        projectionimg=projectionimg.astype(np.uint8)
        cv2.imwrite(path+'/temp6.png',projectionimg)
        
        img_gray=cv2.cvtColor(projectionimg,cv2.COLOR_BGR2GRAY)
        maskindex=np.where(img_gray==0)
        mask=np.zeros(resolution[0:2])
        mask[maskindex]=255
        mask=mask.astype(np.uint8)
        
        # dst= cv2.inpaint(projectionimg,mask,10,cv2.INPAINT_TELEA)
        # cv2.imwrite(path+'/temp7.png',dst)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        dst = cv2.morphologyEx(projectionimg, cv2.MORPH_CLOSE, kernel, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, 1)
        img1 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        
        canny = cv2.Canny(img1, 50, 150)
        canny2=copy.deepcopy(canny)
        lines = cv2.HoughLinesP(canny2,1,np.pi/180,50,minLineLength=30,maxLineGap=100)
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(canny2, (x1, y1), (x2, y2), 255, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        projectionimg2 = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,kernel)
        
        cv2.imwrite(path+'/temp.png',projectionimg)
        cv2.imwrite(path+'/temp2.png',projectionimg2)
        cv2.imwrite(path+'/temp3.png',canny)
        cv2.imwrite(path+'/temp4.png',canny2)
        cv2.imwrite(path+'/temp5.png',dst)
    
    img2 = cv2.cvtColor(projectionimg,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path+'/temp9.png',img2)
    hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
    # plt.plot(hist[1:])
    # plt.show()

    th = threshold(img2, 127)
    print('th', th)
    _,img5 = cv2.threshold(img2, th, 255, cv2.THRESH_TOZERO)
    cv2.imwrite(path+'/temp27.png',img5)

    contours5, hierarchy = cv2.findContours(img5,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    print(len(contours5))
    projectionimg12 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg12, contours5, -1, (0, 0, 255), 1)
    cv2.imwrite(path+'/temp28.png',projectionimg12)

    la=[]
    for i in range(len(contours5)):
        area=cv2.contourArea(contours5[i])
        # area = cv2.arcLength(contours[i], False)
        la.append([i,area])
    la=sorted(la,key=lambda x:x[1],reverse=True)
    cnt3=contours5[la[0][0]]

    projectionimg15 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg15, cnt3, -1, (0, 0, 255), 1)
    cv2.imwrite(path+'/temp30.png',projectionimg15)
    print(cnt3.shape)
    # plt.plot(cnt3[:, 0, 1], cnt3[:, 0, 0])
    # plt.show()
    
    # # cv2多边形拟合方法
    # projectionimg13 = copy.deepcopy(projectionimg)
    # epsilon2 = 0.01*cv2.arcLength(cnt3, True)
    # approx4 = cv2.approxPolyDP(cnt3, epsilon2, True)
    # approx4 = np.reshape(approx4, (approx4.shape[0], approx4.shape[2]))
    # print('1:', approx4, approx4.shape[0])
    # center4 = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), approx4), [len(approx4)]*2))
    # approx4 = np.array(sorted(approx4, key=lambda a: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, a, center4))[::-1]))) % 360, reverse=True))
    # print('2:', approx4, approx4.shape[0])
    # cv2.drawContours(projectionimg13, [approx4], -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp29.png',projectionimg13)

    epsilon3 = 0.01*cv2.arcLength(cnt3, True)
    cnt4 = np.reshape(cnt3, (cnt3.shape[0], cnt3.shape[2]))
    approx5 = thick_polygonal_approximate(cnt4, epsilon3)
    print('3:', approx5, approx5.shape[0])
    projectionimg14 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg14, [approx5], -1, (0, 0, 255), 1)
    cv2.imwrite(path+'/temp32.png',projectionimg14)

    rect2 = cv2.minAreaRect(cnt3)
    box2 = cv2.boxPoints(rect2)  
    box2 = np.int0(box2)
    V2=box2
    projectionimg14 = copy.deepcopy(projectionimg)
    cv2.drawContours(projectionimg14,[V2],-1,[255,0,0],3)  
    cv2.imwrite(path+'/temp31.png',projectionimg14)

    et=time.time()
    tdict['extract edges']=et-st

    return V2, approx5

    # img4 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 10)
    # cv2.imwrite(path+'/temp12.png',img4)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img4 = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, kernel, 1)
    # cv2.imwrite(path+'/temp14.png',img4)

    # img3 = cv2.equalizeHist(img2)
    # cv2.imwrite(path+'/temp10.png',img3)
    # hist2 = cv2.calcHist([img3], [0], None, [256], [0, 256])
    # # plt.plot(hist2[1:])
    # # plt.show()
    
    # #没用
    # contours, hierarchy = cv2.findContours(img3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    # projectionimg3 = copy.deepcopy(projectionimg)
    # cv2.drawContours(projectionimg3, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp11.png',projectionimg3)
    
    # #没用
    # contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # projectionimg6 = copy.deepcopy(projectionimg)
    # cv2.drawContours(projectionimg6, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp8.png',projectionimg6)
    
    # contours, hierarchy = cv2.findContours(img4,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    # projectionimg4 = copy.deepcopy(projectionimg)
    # cv2.drawContours(projectionimg4, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp13.png',projectionimg4)
    
    # projectionimg11 = copy.deepcopy(projectionimg)
    # cv2.drawContours(projectionimg11, contours, -1, (0, 0, 255), -1)
    # cv2.imwrite(path+'/temp26.png',projectionimg11)

    # la=[]
    # for i in range(len(contours)):
    #     area=cv2.contourArea(contours[i])
    #     # area = cv2.arcLength(contours[i], False)
    #     la.append([i,area])
    # la=sorted(la,key=lambda x:x[1],reverse=True)
    # cnt=contours[la[0][0]]

    # projectionimg5 = copy.deepcopy(projectionimg)
    # cv2.drawContours(projectionimg5, cnt, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp15.png',projectionimg5)

    # blankimg = np.zeros(projectionimg.shape)
    # cv2.drawContours(blankimg, cnt, -1, (255, 255, 255), 2)
    # cv2.imwrite(path+'/temp17.png', blankimg)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # erosion = cv2.morphologyEx(blankimg, cv2.MORPH_CLOSE, kernel, 1)
    # cv2.imwrite(path+'/temp18.png', erosion)

    # blankimg1 = cv2.cvtColor(blankimg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # contours3, hierarchy = cv2.findContours(blankimg1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # print(len(contours3))
    # blankimg2 = copy.deepcopy(blankimg)
    # cv2.drawContours(blankimg2, contours3, 0, (255, 255, 255), -1)
    # cv2.imwrite(path+'/temp22.png',blankimg2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # blankimg3 = cv2.morphologyEx(blankimg2, cv2.MORPH_OPEN, kernel, 1)
    # cv2.imwrite(path+'/temp23.png', blankimg3)

    # blankimg4 = cv2.cvtColor(blankimg3.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # contours4, hierarchy = cv2.findContours(blankimg4,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(len(contours4))
    # projectionimg9 = copy.deepcopy(projectionimg)
    # cv2.drawContours(projectionimg9, contours4, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp24.png',projectionimg9)

    # projectionimg10 = copy.deepcopy(projectionimg)
    # epsilon = 0.01*cv2.arcLength(contours4[0], True)
    # approx3 = cv2.approxPolyDP(contours4[0], epsilon, True)
    # print(approx3)
    # print(approx3.shape)
    # approx3 = np.reshape(approx3, (approx3.shape[0], approx3.shape[2]))
    # print(approx3)
    # print(approx3.shape)
    # center3 = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), approx3), [len(approx3)]*2))
    # approx3 = np.array(sorted(approx3, key=lambda a: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, a, center3))[::-1]))) % 360, reverse=True))
    # print(approx3)
    # cv2.drawContours(projectionimg10, [approx3], -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp25.png',projectionimg10)

    # erosion1 = cv2.cvtColor(erosion.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # contours2, hierarchy = cv2.findContours(erosion1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(len(contours2))
    # erosion2 = copy.deepcopy(erosion)
    # cv2.drawContours(erosion2, contours2, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp19.png',erosion2)

    # la=[]
    # for i in range(len(contours2)):
    #     area=cv2.contourArea(contours2[i])
    #     # area = cv2.arcLength(contours[i], False)
    #     la.append([i,area])
    # la=sorted(la,key=lambda x:x[1],reverse=True)
    # cnt2=contours2[la[0][0]]
    
    # erosion3 = copy.deepcopy(erosion)
    # cv2.drawContours(erosion3, cnt2, -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp20.png',erosion3)

    # projectionimg8 = copy.deepcopy(projectionimg)
    # epsilon = 0.01*cv2.arcLength(cnt2, True)
    # approx2 = cv2.approxPolyDP(cnt2, epsilon, True)
    # print(approx2)
    # print(approx2.shape)
    # approx2 = np.reshape(approx2, (approx2.shape[0], approx2.shape[2]))
    # print(approx2)
    # print(approx2.shape)
    # center2 = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), approx2), [len(approx2)]*2))
    # approx2 = np.array(sorted(approx2, key=lambda a: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, a, center2))[::-1]))) % 360, reverse=True))
    # print(approx2)
    # cv2.drawContours(projectionimg8, [approx2], -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp21.png',projectionimg8)

    # projectionimg7 = copy.deepcopy(projectionimg)
    # epsilon = 0.01*cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # print(approx)
    # print(approx.shape)
    # approx = np.reshape(approx, (approx.shape[0], approx.shape[2]))
    # print(approx)
    # print(approx.shape)
    # center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), approx), [len(approx)]*2))
    # approx = np.array(sorted(approx, key=lambda a: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, a, center))[::-1]))) % 360, reverse=True))
    # print(approx)
    # cv2.drawContours(projectionimg7, [approx], -1, (0, 0, 255), 1)
    # cv2.imwrite(path+'/temp16.png',projectionimg7)

    # # print(la)

    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)  
    # box = np.int0(box)
    # V=box

    # # rect_center=np.mean(box,axis=0)
    # # rect_off=box-rect_center

    # # V0=rect_off+rect_center
    # # V0=np.int64(V0)

    # # rect_scale=rect_off*red_coef
    # # V=rect_scale+rect_center
    # # V=np.int64(V)

    # cv2.drawContours(projectionimg,[V],-1,[255,0,0],3)  
    # cv2.imwrite(path+'/edge detection.png',projectionimg)

    # et=time.time()
    # tdict['extract edges']=et-st

    # return approx3, V

def pixel2coords(pcd1, approx, V, mm, dp, margin, tdict):
    print('pixels to coords')

    print(approx)
    print(V)
    print(mm)
    st=time.time()

    minx = mm[0, 0]
    maxy = mm[1, 1]

    # Vertex_app = np.zeros([approx.shape[0], 3])
    # for i in range(approx):
    #     Vertex_app[i] = [(approx[i, 0]+1-margin)*dp+minx, maxy-(approx[i, 1]+1-margin)*dp, 0]
    Vertex_app = np.array([(approx[:, 0]+1-margin)*dp+minx, maxy-(approx[:, 1]+1-margin)*dp, np.zeros(approx.shape[0])]).transpose()
    print(Vertex_app)
    pcdva=o3d.geometry.PointCloud()
    pcdva.points=o3d.utility.Vector3dVector(Vertex_app)

    midx=0
    midy=0
    for i in range(4):
        midx+=V[i][0]/4
        midy+=V[i][1]/4
    midx=(midx+1-margin)*dp+minx
    midy=maxy-(midy+1-margin)*dp
    Vertex=[[],[],[],[]]
    for v in V:
        t=[(v[0]+1-margin)*dp+minx, maxy-(v[1]+1-margin)*dp, 0]
        if t[0]<=midx and t[1]>midy:
            Vertex[0]=t
        elif t[0]>midx and t[1]>midy:
            Vertex[1]=t
        elif t[0]>midx and t[1]<=midy:
            Vertex[2]=t
        elif t[0]<=midx and t[1]<=midy:
            Vertex[3]=t
    pcdv=o3d.geometry.PointCloud()
    pcdv.points=o3d.utility.Vector3dVector(np.array(Vertex))

    #旋转点云，摆正边缘
    ti=[]
    for i in range(4):
        direction=[Vertex[i%4][0]-Vertex[(i+1)%4][0],Vertex[i%4][1]-Vertex[(i+1)%4][1]]
        if direction[0]!=0:
            t=-1*direction[1]/direction[0]
        else:
            t=float('inf')
        ti.append(t)
    
    print(ti)

    theta=math.atan(ti[0])

    pcd2=o3d.geometry.PointCloud()
    xyz1=np.asarray(pcd1.points)
    rgb1=np.asarray(pcd1.colors)
    pcd2.points=o3d.utility.Vector3dVector(xyz1)
    pcd2.colors=o3d.utility.Vector3dVector(rgb1)

    R1=pcd2.get_rotation_matrix_from_zyx((theta,0,0))
    pcd2.rotate(R1,center=(0,0,0))

    xyz2=np.asarray(pcd2.points)

    R2=pcdv.get_rotation_matrix_from_zyx((theta,0,0))
    pcdv.rotate(R2,center=(0,0,0))
    Vertex2=np.asarray(pcdv.points)[:,:2]

    R3=pcdva.get_rotation_matrix_from_zyx((theta,0,0))
    pcdva.rotate(R3,center=(0,0,0))
    Vertex_app2=np.asarray(pcdva.points)[:,:2]

    # 点云采样
    pcd3 = deepcopy(pcd2)
    box = pcd3.get_axis_aligned_bounding_box()
    boxMin = box.get_min_bound()
    boxMax = box.get_max_bound()
    boxMM = np.vstack((boxMin, boxMax))
    print(boxMM)
    # boxP = np.asarray(box.get_box_points())
    # print(boxP)
    # fig=plt.figure()
    # ax=plt.axes(projection='3d')
    # ax.scatter3D(boxP[:, 0], boxP[:, 1], boxP[:, 2])
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()
    diagonal = np.sum(np.power(boxMax-boxMin, 2))**0.5
    print(diagonal)
    downpcd, _, point_map = pcd3.voxel_down_sample_and_trace(voxel_size=diagonal/500, min_bound=boxMin, max_bound=boxMax)
    print(pcd3)
    print(downpcd)
    # o3d.visualization.draw_geometries([downpcd])
    xyzd = np.asarray(downpcd.points)
    rgbd = np.asarray(downpcd.colors)

    # 判断点是否在矩形框内，并剔除框外点
    margin=0
    ind1=np.where((xyzd[:,0]>min(Vertex2[:,0])+margin) & (xyzd[:,0]<max(Vertex2[:,0])-margin))
    temp=xyzd[ind1]
    tempc=rgbd[ind1]
    point_map = [point_map[i] for i in ind1[0].tolist()]
    ind2=np.where((temp[:,1]>min(Vertex2[:,1])+margin) & (temp[:,1]<max(Vertex2[:,1])-margin))
    xyz3=temp[ind2]
    rgb3=tempc[ind2]
    point_map = [point_map[i] for i in ind2[0].tolist()]
    # scalefactor=(np.max(xyz3,axis=0)[0]-np.min(xyz3,axis=0)[0])/100
    # xyz3=xyz3/scalefactor

    # 判断点是否在多边形框内，并剔除框外点
    print(Vertex_app2)
    ind = np.where(inPolygon(Vertex_app2, xyz3[:, :2]))
    xyz4 = xyz3[ind]
    print(xyz4.shape)
    rgb4 = rgb3[ind]
    scalefactor=(np.max(xyz4,axis=0)[0]-np.min(xyz4,axis=0)[0])/100
    xyz4=xyz4/scalefactor
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(xyz4)
    pcd4.colors = o3d.utility.Vector3dVector(rgb4)
    point_map = [point_map[i] for i in ind[0].tolist()]
    print(pcd4)

    pcdMin = pcd4.get_min_bound()
    pcdMax = pcd4.get_max_bound()
    print(pcdMin, pcdMax)
    pcd5, ind3 = pcd4.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    point_map2 = [point_map[i] for i in ind3]
    print(pcd5)

    xyz2 = np.asarray(pcd2.points)
    rgb2 = np.asarray(pcd2.colors)
    xyz2=xyz2/scalefactor
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    pcd2.colors = o3d.utility.Vector3dVector(rgb2)

    # display_inlier_outlier(pcd4, ind3)
    # o3d.visualization.draw_geometries([pcd2, pcd5])

    # o3d.io.write_point_cloud('D:/project/paper_flatten/code/integrated_code/temp/interpcd_4.ply',pcd4,write_ascii= True)

    # xyz3c=xyz2
    # rgb3c=rgb1
    # xyz3c=xyz3c/scalefactor

    # pcd3=o3d.geometry.PointCloud()
    # pcd3.points=o3d.utility.Vector3dVector(xyz3)
    # pcd3.colors=o3d.utility.Vector3dVector(rgb3)

    # cl, ind = pcd3.remove_statistical_outlier(nb_neighbors=60,std_ratio=5)
    # pcd3 = pcd3.select_by_index(ind)

    et=time.time()
    tdict['pixels to coords']=et-st

    return pcd2, pcd5, point_map2

# def mesh_sampling(pcd, red_coef, red_coef2, tdict):
#     print('mesh sampling')
#     st=time.time()

#     xyz=np.asarray(pcd.points)
#     rgb=np.asarray(pcd.colors)
    
#     dx=max(xyz[:,0])-min(xyz[:,0])
#     dy=max(xyz[:,1])-min(xyz[:,1])
#     xlim=[min(xyz[:,0])-0.5*red_coef2*dx*(1/red_coef-1),max(xyz[:,0])+0.5*red_coef2*dx*(1/red_coef-1)]
#     ylim=[min(xyz[:,1])-0.5*red_coef2*dy*(1/red_coef-1),max(xyz[:,1])+0.5*red_coef2*dy*(1/red_coef-1)]

#     ind1=np.where((xyz[:,0]>=xlim[0])&(xyz[:,0]<xlim[1]))
#     xyzt=xyz[ind1]
#     rgbt=rgb[ind1]
#     ind2=np.where((xyzt[:,1]>=ylim[0])&(xyzt[:,1]<ylim[1]))
#     xyznm=xyzt[ind2]
#     rgbnm=rgbt[ind2]
#     pcdnm=o3d.geometry.PointCloud()
#     pcdnm.points=o3d.utility.Vector3dVector(xyznm)
#     pcdnm.colors=o3d.utility.Vector3dVector(rgbnm)

#     print(pcd)
#     print(pcdnm)
#     print((xyz==xyznm).all())
#     print((rgb==rgbnm).all())

#     et=time.time()
#     tdict['mesh sampling']=et-st

#     return pcdnm

def save(path, pcd_ori, pcd_ds, point_map, tdict):
    print('saving')
    st=time.time()

    o3d.io.write_point_cloud(path+'/interpcd_ori.ply',pcd_ori,write_ascii= True)
    o3d.io.write_point_cloud(path+'/interpcd_downsample.ply',pcd_ds,write_ascii= True)
    # o3d.io.write_point_cloud(path+'/interpcd_nomesh.ply',pcdnm,write_ascii= True)

    maps = pd.DataFrame(data=point_map, index=None, columns=None)
    maps.to_csv(path+'/point_map.csv')

    et=time.time()
    tdict['saving']=et-st

    print('pre processing done')

    # o3d.visualization.draw_geometries([pcd_ori],mesh_show_wireframe=True,mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([pcd_ds],mesh_show_wireframe=True,mesh_show_back_face=True)

def preprocess(**kwargs):
    print('begin')

    filename = kwargs["input_pcd"]
    path = kwargs["work_path"]

    if not os.path.exists(path):
        os.makedirs(path)

    tdict = dict()
    margin=50
    # red_coef=0.9
    # red_coef2=0.8

    pcdo, plytype = load_pcd(filename, tdict)
    pcd1 = rigid_transformation(pcdo, tdict)
    projectionimg, mm, dp = projection(pcd1, plytype, margin, tdict)
    V, approx = edge_extraction(path, pcd1, plytype, projectionimg, mm, dp, margin, tdict)
    pcd_ori, pcd_ds, point_map = pixel2coords(pcd1, approx, V, mm, dp, margin, tdict)
    # pcdnm = mesh_sampling(pcd_ds, red_coef, red_coef2, tdict)
    save(path, pcd_ori, pcd_ds, point_map, tdict)
    
    print(str(tdict))

    return pcd_ds, pcd_ori, point_map

if __name__ == '__main__':
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    # kwargs = {"input_pcd": r'D:\project\paper_flatten\data\case_002.ply', 
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}
    kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_Gray/one_way_bending/case_002/case_002.ply', 
            "work_path": current_path + '/temp', 
            "output_path": current_path + '/result'}
    # kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_Gray/a_mess/case_008/case_008.ply', 
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}
    # kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_Gray/multi_directional_bending/case_004/case_004.ply', 
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}
    # kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_RGB/a_mess/case_005/case_005.ply', 
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}
    # kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_RGB/a_mess/case_003/case_003.ply', 
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}
    # kwargs = {"input_pcd": current_path + '/ply/DenseCloud.ply', 
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}

    pcd_ds, pcd_ori, point_map = preprocess(**kwargs)
    print(pcd_ds)
