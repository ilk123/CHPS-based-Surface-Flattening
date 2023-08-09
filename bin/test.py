import os
import re
import shutil
import open3d as o3d
import numpy as np
import cv2

# from source1_0403 import preprocess
from flatten import flatten
from evaluation import evalSingle

path = "D:/project/paper_flatten/data/yaguan"

def get_filelist(dir, type):
    filelist = []

    for home, _, files in os.walk(dir):
        for filename in files:
            if re.match(type, filename) is not None:
                filelist.append(os.path.join(home, filename))

    return filelist

if __name__ == '__main__':
    sizeList = []
    
    plyList = get_filelist(path, 'case_....ply')
    oriImgList = get_filelist(path, 'original.jpg')

    print(len(plyList), len(oriImgList))

    for ply, img in zip(plyList, oriImgList):
        print(ply, img)

    #     pcd = o3d.io.read_point_cloud(file)
    #     point = np.asarray(pcd.points)
    #     sizeList.append(point.size)
    
    # sizeList = np.array(sizeList)
    # print(sizeList.max(), sizeList.min(), sizeList.mean())

    g_path = 'D:/project/paper_flatten/code/integrated_code/result/all_case'
    if not os.path.exists(g_path):
        os.mkdir(g_path)
    with open(g_path + '/error.txt', 'w') as f, open(g_path + '/result.txt', 'w') as r:
        f.truncate()
        r.truncate()

        for i, File in enumerate(zip(plyList, oriImgList)):
            file = File[0]
            oriFile = File[1]
            shutil.copyfile(oriFile, g_path+'/{}_ori.jpg'.format(i))
            temp = file.split('\\')
            path = ''
            for j in range(len(temp)-1):
                path += temp[j]
                path += '\\'
            kwargs = {"input_pcd": file, 
                    "work_path": path + 'temp', 
                    "output_path": path + 'result'}
            if not os.path.exists(kwargs['work_path']):
                os.makedirs(kwargs['work_path'])
            if not os.path.exists(kwargs['output_path']):
                os.makedirs(kwargs['output_path'])
            try:
                # pcdnm = preprocess(**kwargs)
                pcdo, dst, img = flatten(**kwargs)
                oriImg = cv2.imread(oriFile)
                cv2.imwrite(g_path+'/{}_dst.jpg'.format(i), dst)
                cv2.imwrite(g_path+'/{}_img.jpg'.format(i), img)
                resized_img, gd, ld, ssim, ed, cer = evalSingle(dst, oriImg)
                cv2.imwrite(g_path+'/{}.jpg'.format(i), resized_img)
                print('shape: ', dst.shape, img.shape, resized_img.shape)
                r.write(str(i) + ': ')
                r.write(kwargs['input_pcd'])
                r.write('\ngd: %.4f, ld: %.4f, ssim: %.4f, ed: %.4f, cer: %.4f\n' % (gd, ld, ssim, ed, cer))
                r.flush()
            except Exception as e:
                print(kwargs['input_pcd'] + ' can not be processed because ', e)
                f.write(kwargs['input_pcd'])
                f.write(' can not be processed because ' + str(e) + '\n')
                f.flush()
