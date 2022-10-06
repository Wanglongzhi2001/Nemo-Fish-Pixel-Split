# Nemo-Fish-Pixel-Split
## 模式识别与机器学习第一次大作业：给定5张Nemo鱼的图片完成下图所示Nemo鱼黄白像素的聚类分割效果
### 给定条件: array_sample.mat和Mask.mat，其中array_sample.mat存放着309.bmp的像素值(但是图像数据被展平，无位置信息)，Mask.mat为309.bmp中小鱼的掩膜，将其与原图像相乘可分割小鱼和背景，但是其他四张图片未给出掩膜。
![gray_segment_7](https://user-images.githubusercontent.com/69797242/194232505-d76344ea-8086-49ab-9ac6-d522ed4e2018.jpg)
![RGB_segment_20](https://user-images.githubusercontent.com/69797242/194232743-ce02090c-d156-4d4c-9737-dc61ce6a9878.jpg)
### 任务一：给定的五幅图像上测试，其中309.bmp既考虑灰度图像又考虑彩色图像
### 任务二：去掉给定的label信息，基于EM算法实现彩色图像(309.bmp)的分割



