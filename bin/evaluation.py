import os
import numpy as np
import cv2
import pytesseract
import matlab.engine as matlab

def global_distortion(img, oriImg):
    img = cv2.resize(img, dsize=(oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(oriImg, None)
    # kp_oriImg = cv2.drawKeypoints(oriImg, kp1, None)
    # kp_img = cv2.drawKeypoints(img, kp2, None)
    # cv2.imshow('1', kp_img)
    # cv2.imshow('2', kp_oriImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    ratio = 0.85
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])
    # matches = cv2.drawMatchesKnn(oriImg, kp1, img, kp2, good_matches, None, flags=2)
    # cv2.imshow('1', matches)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    if len(good_matches) > 4:
        ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        # print(ptsA.shape, ptsA[0:3], '\n', ptsB.shape, ptsB[0:3])

        A = cv2.getAffineTransform(ptsA[:3], ptsB[:3])
        # print(A)
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        # print(H)
        # imgOut = cv2.warpPerspective(img, H, (oriImg.shape[1], oriImg.shape[0]), flag=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    G = (A[0][0]*A[1][1]-A[1][0]*A[0][1])
    G = max(G, 1/G)
    # print(G)
    J = (H[0][0]*H[1][1]-H[1][0]*H[0][1])
    J = max(J, 1/J)
    print('gd: ', J)

    return img, A, H, J

def SSIM_local_distortion(img, oriImg, H):
    corrImg = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep)
    eng = matlab.start_matlab()
    eng.cd(current_path,nargout=0)
    ms, ld = eng.ssim_ld_eval(corrImg, oriImg, nargout=2)
    print('ld: ', ld)
    print('ssim: ', ms)
    return ms, ld

def Levenshtein_Distance(str1, str2):
    matrix = [[i+j for j in range(len(str2)+1)] for i in range(len(str1)+1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            d = 0 if str1[i-1] == str2[j-1] else 0
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    return matrix[len(str1)][len(str2)]

def cal_cer_ed(img, oriImg):
    # path_gt='./GT/'
    # gt=Image.open(path_gt+str(i)+'.png')
    # img1=Image.open(path_ours+str(i)+'_1' + tail)
    # img2=Image.open(path_ours+str(i)+'_2' + tail)
    # gt=Image.open(path_ours+str(i)+'.png')
    # img1=Image.open(path_ours+str(i)+'_1'+'.png')

    content_gt=pytesseract.image_to_string(oriImg)
    content1=pytesseract.image_to_string(img)
    ed=Levenshtein_Distance(content_gt,content1)
    cer = ed/len(content_gt)
    
    print('CER: ', cer)
    print('ED:  ', ed)
    return cer, ed

def evalSingle(img, oriImg):
    img, _, H, gd = global_distortion(img, oriImg)
    ssim, ld = SSIM_local_distortion(img, oriImg, H)
    cer, ed = cal_cer_ed(img, oriImg)
    return img, gd, ld, ssim, ed, cer
