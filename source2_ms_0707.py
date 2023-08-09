import os
import copy
import time
import pandas as pd
import math

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import trange

from retinex import MSRCR
from lama.bin import inpaint

def reverse(pcd, ind_nan):
    n = np.asarray(pcd.points).shape[0]
    ind = []
    for i in range(n):
        if i not in ind_nan:
            ind.append(i)
    ind = np.array(ind)
    return ind

def map_cleansing(map, ind_nan):
    # cmap = []
    # for i in range(len(map)):
    #     if i in ind_nan:
    #         temp = [int(j) for j in map[i] if not math.isnan(j)]
    #         cmap.append(temp)
    # return cmap
    for i in range(len(map)):
        map[i] = [int(j) for j in map[i] if not math.isnan(j)]
    return map

def map_flatten(map):
    fmap = []
    for i in range(len(map)):
        inv = np.asarray(map[i])
        for j in range(len(inv)):
            fmap.append(inv[j])
    fmap = np.sort(np.asarray(fmap, dtype=int))
    print(fmap.shape)
    return fmap

def projection(pcdo):
    xyz = np.asarray(pcdo.points)
    rgb = np.asarray(pcdo.colors)

    margin=10
    minx=np.min(xyz[:,0])
    maxx=np.max(xyz[:,0])
    miny=np.min(xyz[:,1])
    maxy=np.max(xyz[:,1])
    dx=maxx-minx
    dy=maxy-miny

    print(xyz)
    print(maxx, maxy, minx, miny)
    ratio=dx/dy
    print(len(xyz), ratio)
    resx=int((len(xyz)*1/ratio)**0.5*ratio)
    resy=int((len(xyz)*1/ratio)**0.5)
    resolution=[resy,resx,3]

    if ratio>=(resolution[1]-2*margin)/(resolution[0]-2*margin):
        dp=dx/(resolution[1]-2*margin)
    else:
        dp=dy/(resolution[0]-2*margin)
    projectionimg=np.zeros(resolution)
    projectionimg=projectionimg.astype(np.uint8)

    label_list = np.zeros(projectionimg.shape[:2])
    for i in range(len(xyz)):
        lx = int((xyz[i][0]-minx)/dp)-1
        ly = int((maxy-xyz[i][1])/dp)-1
        if not label_list[ly, lx]:
            c=rgb[i]*255
            c=[c[2],c[1],c[0]]
            projectionimg[ly+margin][lx+margin] = c
            label_list[ly][lx] = 1
    projectionimg = cv2.flip(projectionimg, 1)

    return projectionimg


def imgInpaint(projectionimg):
    resolution = projectionimg.shape

    projectionimg=projectionimg.astype(np.uint8) 

    img_gray = cv2.cvtColor(projectionimg,cv2.COLOR_BGR2GRAY)
    maskindex=np.where(img_gray==0)
    mask=np.zeros(resolution[:-1])
    mask[maskindex]=255
    mask=mask.astype(np.uint8)

    # dst=cv2.inpaint(projectionimg,mask,7,cv2.INPAINT_TELEA)
    projectionimg_float = projectionimg.astype('float32') / 255
    mask_float = mask.astype('float32') / 255
    dst = inpaint(projectionimg_float, mask_float)

    return dst

def illumination(dst):
    img=copy.deepcopy(dst)

    img_msrcr = MSRCR(img,sigma_list=[15, 80, 250],
                    G=192.0, b=-30.0,
                    alpha=125.0, beta=46.0,
                    low_clip=0.01, high_clip=0.8)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_r_img = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2HSV)

    v_r_img = hsv_r_img[:, :, 2] + 20
    v_r_img[v_r_img>255] = 255

    hsv_m_img1 = np.dstack((hsv_img[:, :, 0], hsv_img[:, :, 1], v_r_img))
    m_img1 = cv2.cvtColor(hsv_m_img1, cv2.COLOR_HSV2BGR)

    return img_msrcr, m_img1


def postprocess(pcd_ori, pcd_ds, pcdfi, point_map, ind_nan, **kwargs):
    print('postprocess')
    time0 = time.time()

    path = kwargs["output_path"]

    if not os.path.exists(path):
        os.makedirs(path)

    xyz0 = np.asarray(pcd_ori.points)
    rgb0 = np.asarray(pcd_ori.colors)
    xyz1 = np.asarray(pcd_ds.points)
    xyz2 = np.asarray(pcdfi.points)
    print(np.count_nonzero(np.isnan(xyz2[:,0])))
    # print(pcd_ds, pcdfi)
    dis = xyz2 - xyz1
    for i in range(len(point_map)):
        xyz0[point_map[i]] += dis[i]
    ind = [point_map[i] for i in range(len(point_map)) if i not in ind_nan]
    ind = [j for i in ind for j in i]
    print(len(ind))
    # ind = map_flatten(point_map)
    xyz0 = xyz0[ind]
    rgb0 = rgb0[ind]
    pcdo = o3d.geometry.PointCloud()
    pcdo.points = o3d.utility.Vector3dVector(xyz0)
    pcdo.colors = o3d.utility.Vector3dVector(rgb0)
    print(pcdo)
    o3d.visualization.draw_geometries([pcdo], 'pcdo2')

    # pcdr=o3d.geometry.PointCloud()
    # xyz1 = np.asarray(pcdfi.points)
    # rgb1 = np.asarray(pcd_ds.colors)
    # pcdr.points = o3d.utility.Vector3dVector(xyz1)
    # pcdr.colors = o3d.utility.Vector3dVector(rgb1)
    # # pcdr.points=pcdfi.points
    # # pcdr.colors=pcdnm.colors
    # # xyz1=np.array(pcdr.points)
    # # rgb1=np.array(pcdr.colors)
    # xyz2 = np.asarray(pcdr.points)
    # print(xyz1 == xyz2)
    # print(xyz1.shape, rgb1.shape)

    projectionimg = projection(pcdo)

    dst = imgInpaint(projectionimg)

    img_msrcr, m_img1 = illumination(dst)

    cv2.imwrite(path+'/output.jpg',projectionimg)
    cv2.imwrite(path+'/output2.jpg',dst)
    cv2.imwrite(path+'/output3.jpg',img_msrcr)
    cv2.imwrite(path+'/output4.jpg',m_img1)

    o3d.io.write_point_cloud(path+'/result.ply',pcdo,write_ascii= True)
    
    # o3d.visualization.draw_geometries([pcdo])

    print(time.time()-time0)

    return pcdo, dst, m_img1

if __name__ == '__main__':
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    kwargs = {"input_pcd": current_path + '/ply/DenseCloud.ply', 
            "work_path": current_path + '/temp', 
            "output_path": current_path + '/result'}
    # kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_Gray/one_way_bending/case_002/case_002.ply', >
    #         "work_path": current_path + '/temp', 
    #         "output_path": current_path + '/result'}

    pcd_ori=o3d.io.read_point_cloud(kwargs['work_path']+'/interpcd_ori.ply')
    pcd_ds=o3d.io.read_point_cloud(kwargs['work_path']+'/interpcd_downsample.ply')
    pcdfi=o3d.io.read_point_cloud(kwargs['work_path']+'/interpcd_f.ply')
    print(np.asarray(pcd_ori.points).shape)

    ind_nan = pd.read_csv(kwargs['work_path']+'/ind_nan.csv', index_col=0).values.flatten()
    print('1', len(ind_nan))
    ind_nan = reverse(pcdfi, ind_nan)

    point_map = pd.read_csv(kwargs['work_path']+'/point_map.csv', index_col=0).values.tolist()
    print(len(point_map))
    point_map = map_cleansing(point_map, ind_nan)

    # o3d.visualization.draw_geometries([pcd_ds], window_name="downsample", width=800, height=600)
    
    pcdo, dst, m_img1 = postprocess(pcd_ori, pcd_ds, pcdfi, point_map, ind_nan, **kwargs)
