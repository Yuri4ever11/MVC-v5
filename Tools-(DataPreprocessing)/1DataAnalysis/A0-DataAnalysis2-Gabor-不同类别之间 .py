import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel

# 设置图片路径
img_dir = './output/loose_s'

# 读取图像文件
img_files = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, f) for f in img_files]
images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

# 定义Gabor滤波器参数
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for freq in (0.05, 0.1, 0.15, 0.2):
        for sigma in (1, 2, 3, 4):
            kernel = np.real(gabor_kernel(frequency=freq,
                                         theta=theta,
                                         sigma_x=sigma,
                                         sigma_y=sigma))
            kernels.append(kernel)

# 计算Gabor纹理特征
features = []
for img in images:
    img_features = np.zeros((len(kernels), 2), dtype=np.double)
    for k_idx, kernel in enumerate(kernels):
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        img_features[k_idx, 0] = filtered.mean()
        img_features[k_idx, 1] = filtered.var()
    features.append(img_features.flatten())

# 将特征转换为numpy数组
features = np.array(features)

# 输出特征统计信息
print("Gabor Texture Feature Statistics:")
print("Mean:", features[:, ::2].mean(axis=0))
print("Variance:", features[:, 1::2].mean(axis=0))