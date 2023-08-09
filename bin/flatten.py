import os
import numpy as np
import cv2

from source1_0403 import preprocess
from column_spring_mass_1 import flattening
from source2_ms_0707 import postprocess
from evaluation import evalSingle

def flatten(**kwargs):
    pcd_ds, pcd_ori, point_map = preprocess(**kwargs)
    pcdfi, ind_nan = flattening(pcd_ds, **kwargs)
    pcdo, dst, img = postprocess(pcd_ori, pcd_ds, pcdfi, point_map, ind_nan, **kwargs)

    return pcdo, dst, img

if __name__ == '__main__':
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    kwargs = {"input_pcd": current_path + '/ply/case_001.ply', 
            "work_path": current_path + '/temp', 
            "output_path": current_path + '/result'}
#     kwargs = {"input_pcd": 'D:/project/paper_flatten/data/yaguan/data_Gray/one_way_bending/case_002/case_002.ply', 
#             "work_path": current_path + '/temp', 
#             "output_path": current_path + '/result'}

#     pcd_ds, pcd_ori, point_map = preprocess(**kwargs)
#     pcdfi, ind_nan = flattening(pcd_ds, **kwargs)
#     print('point_map', len(point_map))
#     print('ind_nan', len(ind_nan))
#     pcdr, dst, m_img1 = postprocess(pcd_ori, pcd_ds, pcdfi, point_map, ind_nan, **kwargs)

    pcdo, dst, img = flatten(**kwargs)
    oriImg = cv2.imread(kwargs['output_path']+'/original.jpg')
    img = cv2.imread(kwargs['output_path'] + '/output2.jpg')
    print(img.shape)
    img, gd, ld, ssim, ed, cer = evalSingle(img, oriImg)
