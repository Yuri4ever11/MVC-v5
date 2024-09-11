import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# 设置图片路径
img_dir = './output/loose_s'

# 读取图像文件
img_files = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, f) for f in img_files]
images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

# 计算GLCM纹理特征
features = []
for img in images:
    # 计算GLCM矩阵
    glcm = graycomatrix(img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

    # 提取纹理特征
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-7))

    features.append([contrast, correlation, entropy])

# 将特征转换为numpy数组
features = np.array(features)

# 输出特征统计信息
print("GLCM Texture Feature Statistics:")
print("Contrast:", features[:, 0].mean())
print("Correlation:", features[:, 1].mean())
print("Entropy:", features[:, 2].mean())