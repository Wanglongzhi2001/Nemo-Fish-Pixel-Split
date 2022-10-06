'''
Author: wangdapao666 583087864@qq.com
Date: 2022-09-23 23:37:30
LastEditors: wangdapao666 583087864@qq.com
LastEditTime: 2022-09-28 22:56:22
FilePath: \Python\split_fish.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import cv2
import numpy as np
import os

def split_fish(ori_img, save_img_path,sava_img_name):
    # cv2.imshow('ori_img',ori_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # K均值聚类
    image_2D = ori_img.reshape(ori_img.shape[0]*ori_img.shape[1], ori_img.shape[2]) # 先要将图片reshape才能送进KMeans
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, max_iter=300,random_state=0).fit(image_2D) # K=3
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    # 2D->3D
    clustered_3D = clustered.reshape(ori_img.shape[0], ori_img.shape[1], ori_img.shape[2]).astype(np.uint8)
    # cv2.imshow('after_cluster', clustered_3D)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 灰度化
    cluster_gray = cv2.cvtColor(clustered_3D,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('cluster_gray', cluster_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 阈值化
    cluster_gray[(cluster_gray>100)] = 255
    # cv2.imshow('gray_thresh', cluster_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # 二值化
    ret, thresh = cv2.threshold(cluster_gray, 127, 255, cv2.THRESH_BINARY)
    # 寻找轮廓(有多个，要进行筛选)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 计算平均面积
    areas = list()
    idx = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i], False)
        areas.append(area)
        if abs(area - 6733)<800: # 发现5张图小鱼面积基本就在6000~7000，所以在所有轮廓的面积中找在这个范围的
            idx = i
        #print("轮廓%d的面积:%d" % (i, area))

    # 画轮廓
    img_temp = np.zeros(ori_img.shape, np.uint8)
    fish_contour = cv2.drawContours(img_temp, contours, idx, (0, 0, 255), 1)
    # cv2.imshow("contour", img_temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 填充轮廓
    result = cv2.fillConvexPoly(fish_contour, contours[idx], (255, 255, 0))
    # cv2.imshow("contour_filled", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 转灰度
    res_gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    res_gray[(res_gray > 10)] = 255

    # cv2.imshow("final_result", res_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 在当前文件夹下生成一张split_result图片
    cv2.imwrite(f'./{save_img_path}/{save_img_name}.jpg', res_gray)

if __name__ == '__main__':
    # 分割结果保存在当前目录下的split_result文件夹下
    split_result_dir = 'split_result'
    if not os.path.exists(split_result_dir):
        os.mkdir(split_result_dir)
    for i in range(309, 319, 2):
        ori_img = cv2.imread(f'./data/{i}.bmp')
        save_img_name = 'split_result' + str(i)
        split_fish(ori_img, split_result_dir,save_img_name)
        print(f'{i}.bmp已分割并保存')