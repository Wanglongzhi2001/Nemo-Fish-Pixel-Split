import os
import scipy.io as sio
from scipy.stats import norm, multivariate_normal
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


def EM(src_img_name, mask_name, result_dir):
    # 分割结果放在当前路径下的EM_result文件夹
    result_dir = result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print(f'------结果已存放在当前目录下的{result_dir}文件夹------')
    # 数读取数据
    mask = np.array(Image.open(mask_name)) / 255.


    array_sample = sio.loadmat('./data/array_sample.mat')['array_sample']
    src_img = Image.open(src_img_name)
    RGB_img = np.array(src_img)
    Gray_img = np.array(src_img.convert('L'))

    # 乘上mask获取感兴趣ROI区域
    Gray_ROI = (Gray_img * mask)/255
    RGB_mask = np.array([mask, mask, mask]).transpose(1,2,0) #将c*h*w转换成h*w*c
    RGB_ROI = (RGB_img * RGB_mask)/255

    """初始设定"""
    # 先验概率相同
    P_pre1 = 0.5
    P_pre2 = 0.5

    # 每个数据来自两类的初始概率相同，即软标签相同
    soft_guess1 = 0.5
    soft_guess2 = 0.5


    """EM分割灰度图像"""
    print('------开始进行灰度分割------')
    # 均值和标准差初始设定
    gray1_m = 0.5
    gray1_s = 0.1
    gray2_m = 0.8
    gray2_s = 0.3

    # 绘制初始的PDF
    x = np.arange(0, 1, 1/1000)
    gray1_pdf = norm.pdf(x, gray1_m, gray1_s)
    gray2_pdf = norm.pdf(x, gray2_m, gray2_s)
    show_pic = True # 是否展示原始PDF和ROI区域
    if show_pic:
        plt.figure(0)
        ax = plt.subplot(1, 1, 1)
        ax.plot(x, gray1_pdf, 'r', label='original pdf of first class')
        ax.plot(x, gray2_pdf, 'b', label='original pdf of second class')
        ax.set_title('Original PDF')
        plt.figure(1)
        ax1 = plt.subplot(1, 1, 1)
        ax1.imshow(Gray_ROI, cmap='gray')
        ax1.set_title('Gray ROI')

    gray = np.zeros((len(array_sample), 5)) #用来存放原始灰度数据、类别1和0的个数、类别1和0的样本
    gray_s_old = gray1_s + gray2_s

    epoch = 30
    # 迭代更新参数
    for epoch in range(epoch):
        """E步骤(贝叶斯得到后验概率)"""
        for i in range(len(array_sample)):
            soft_guess1 = (P_pre1*norm.pdf(array_sample[i][0], gray1_m, gray1_s))/(P_pre1*norm.pdf(array_sample[i][0], gray1_m, gray1_s) +
                                                                             P_pre2*norm.pdf(array_sample[i][0], gray2_m, gray2_s))
            soft_guess2 = 1 - soft_guess1
            gray[i][0] = array_sample[i][0]
            gray[i][1] = 0 if soft_guess1 < 0.5 else 1               # 当前一个数据中类别1占的个数
            gray[i][2] = 0 if soft_guess2 < 0.5 else 1
            gray[i][3] = soft_guess1*array_sample[i][0]              # 对当前数据中属于类别1的部分，当前数据*后验
            gray[i][4] = soft_guess2*array_sample[i][0]

        """M步骤，由最大似然估计计算PDF的参数(均值、标准差)"""
        gray1_num = sum(gray)[1]                                # 类别1的总数
        gray2_num = sum(gray)[2]
        gray1_m = sum(gray)[3]/gray1_num                        # 类别1的均值
        gray2_m = sum(gray)[4]/gray2_num

        sum_s1 = 0.
        sum_s2 = 0.

        for i in range(len(gray)):
            sum_s1 += gray[i][1] * gray[i][1]*(gray[i][0] - gray1_m)*(gray[i][0] - gray1_m)  # 每个数据的波动中，属于类别1的部分
            sum_s2 += gray[i][2] * gray[i][2]*(gray[i][0] - gray2_m)*(gray[i][0] - gray2_m)
        gray1_s = np.power(sum_s1/(gray1_num-1), 0.5)   # 标准差的无偏估计
        gray2_s = np.power(sum_s2/(gray2_num-1), 0.5)

        P_pre1 = gray1_num/(gray1_num + gray2_num)                                         # 更新先验概率
        P_pre2 = 1 - P_pre1

        gray1_pdf = norm.pdf(x, gray1_m, gray1_s)
        gray2_pdf = norm.pdf(x, gray2_m, gray2_s)
        gray_s_d = abs(gray_s_old - gray2_s - gray1_s)
        gray_s_old = gray2_s + gray1_s

        # 绘制更新参数后的pdf
        plt.figure(2)
        ax2 = plt.subplot(1, 1, 1)
        ax2.plot(x, gray1_pdf, 'r', label='pdf of first class')
        ax2.plot(x, gray2_pdf, 'b', label='pdf of seconf class')
        ax2.set_title(f'epoch {epoch + 1} PDF')
        ax2.legend()

        plt.savefig(f'{result_dir}//PDF_{str(epoch + 1)}.jpg')
        plt.close()


        gray_out = np.zeros_like(Gray_ROI, dtype=np.uint8)
        for i in range(Gray_ROI.shape[0]):
            for j in range(Gray_ROI.shape[1]):
                if abs(Gray_ROI[i][j] - 0.) < 1e-2:
                    continue
                # 贝叶斯决策
                elif P_pre1 * norm.pdf(Gray_ROI[i][j], gray1_m, gray1_s) > P_pre2 * norm.pdf(Gray_ROI[i][j], gray2_m, gray2_s):
                    gray_out[i][j] = 100
                elif P_pre1 * norm.pdf(Gray_ROI[i][j], gray1_m, gray1_s) <= P_pre2 * norm.pdf(Gray_ROI[i][j], gray2_m, gray2_s):
                    gray_out[i][j] = 255
        # 显示分割结果
        plt.figure(3)
        ax3 = plt.subplot(1, 1, 1)
        ax3.imshow(gray_out, cmap='gray')

        ax3.set_title(f'epoch{epoch + 1} gray segment')
        plt.savefig(f'{result_dir}//gray_segment_{str(epoch + 1)}.jpg')
        print(f'---第{epoch + 1}轮灰度分割结果图片已保存---')
        # plt.show()
        plt.close()

        if gray_s_d < 0.0001:       # 停止迭代
            break


    """EM分割RGB图像"""
    print('------开始进行RGB分割------')
    # 均值和协方差初始设定
    RGB1_m = np.array([0.45, 0.45, 0.45])
    RGB2_m = np.array([0.8, 0.8, 0.8])
    RGB1_cov = np.array([[0.1, 0.05, 0.04],
                        [0.05, 0.1, 0.02],
                        [0.04, 0.02, 0.1]])
    RGB2_cov = np.array([[0.1, 0.05, 0.04],
                        [0.05, 0.1, 0.02],
                        [0.04, 0.02, 0.1]])

    RGB = np.zeros((len(array_sample), 11))

    # 显示彩色ROI
    show_pic = False
    if show_pic:
        plt.figure(3)
        ax3 = plt.subplot(1, 1, 1)
        ax3.set_title('RGB ROI')
        ax3.imshow(RGB_ROI)
        plt.show()
        plt.close()
    # 迭代更新参数
    epoch = 20
    for epoch in range(epoch):
        for i in range(len(array_sample)):

            # 贝叶斯计算每个数据的后验，即得到软标签
            soft_guess1 = P_pre1 * multivariate_normal.pdf(array_sample[i][1:4], RGB1_m, RGB1_cov)/(P_pre1 * \
            multivariate_normal.pdf(array_sample[i][1:4], RGB1_m, RGB1_cov) + P_pre2 * multivariate_normal.pdf(array_sample[i][1:4], RGB2_m, RGB2_cov))
            soft_guess2 = 1 - soft_guess1
            RGB[i][0:3] = array_sample[i][1:4]
            RGB[i][3] = 0 if soft_guess1 < 0.5 else 1
            RGB[i][4] = 0 if soft_guess2 < 0.5 else 1
            RGB[i][5:8] = soft_guess1 * array_sample[i][1:4]
            RGB[i][8:11] = soft_guess2 * array_sample[i][1:4]
        # print(RGB[0])

        # 根据软标签，再借助最大似然估计出类条件概率PDF参数——均值，标准差
        RGB1_num = sum(RGB)[3]
        RGB2_num = sum(RGB)[4]
        RGB1_m = sum(RGB)[5:8] / RGB1_num
        RGB2_m = sum(RGB)[8:11] / RGB2_num

        # print(RGB1_num+RGB2_num, RGB1_m, RGB2_m)
        cov_sum1 = np.zeros((3, 3))
        cov_sum2 = np.zeros((3, 3))

        for i in range(len(RGB)):
            # print(np.dot((RGB[i][0:3]-RGB1_m).reshape(3, 1), (RGB[i][0:3]-RGB1_m).reshape(1, 3)))
            cov_sum1 = cov_sum1 + RGB[i][3] * np.dot((RGB[i][0:3]-RGB1_m).reshape(3, 1), (RGB[i][0:3]-RGB1_m).reshape(1, 3))
            cov_sum2 = cov_sum2 + RGB[i][4] * np.dot((RGB[i][0:3]-RGB2_m).reshape(3, 1), (RGB[i][0:3]-RGB2_m).reshape(1, 3))
        RGB1_cov = cov_sum1/(RGB1_num-1)   # 协方差无偏估计
        RGB2_cov = cov_sum2/(RGB2_num-1)

        P_pre1 = RGB1_num/(RGB1_num + RGB2_num)
        P_pre2 = 1 - P_pre1

        # 用贝叶斯对彩色图像进行分割

        RGB_out = np.zeros_like(RGB_ROI)

        for i in range(RGB_ROI.shape[0]):
            for j in range(RGB_ROI.shape[1]):
                if abs(np.sum(RGB_ROI[i][j]) - 0.) < 1e-2:
                    continue
                # 贝叶斯决策
                elif P_pre1 * multivariate_normal.pdf(RGB_ROI[i][j], RGB1_m, RGB1_cov) > P_pre2 * multivariate_normal.pdf(
                        RGB_ROI[i][j], RGB2_m, RGB2_cov):
                    RGB_out[i][j] = [255, 0, 0]
                elif P_pre1 * multivariate_normal.pdf(RGB_ROI[i][j], RGB1_m, RGB1_cov) <= P_pre2 * multivariate_normal.pdf(
                        RGB_ROI[i][j], RGB2_m, RGB2_cov):
                    RGB_out[i][j] = [255, 255, 255]
        # print(RGB_ROI.shape)

        # 显示彩色分割结果
        plt.figure(4)
        ax4 = plt.subplot(1, 1, 1)
        ax4.imshow(RGB_out)
        ax4.set_title(f'epoch{epoch + 1} RGB segment')
        plt.savefig(f'{result_dir}//RGB_segment_{str(epoch + 1)}.jpg')
        print(f'---第{epoch + 1}轮RGB分割结果图片已保存---')
        # plt.show()
        plt.close()
if __name__ == '__main__':
    for i in range(309, 319, 2):
        src_img_name = './data/' + str(i) + '.bmp'
        print(f'------开始进行{src_img_name}的EM算法分割------')
        mask_name = './split_result/split_result' + str(i) + '.jpg'
        result_dir = 'EM_result' + str(i)
        EM(src_img_name, mask_name, result_dir)
    print('程序运行结束！！！')