# import os
# import cv2
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
#
# # 设置图片路径
# img_dir = './output/loose_s'
#
# # 读取图像文件
# img_files = os.listdir(img_dir)
# img_paths = [os.path.join(img_dir, f) for f in img_files]
# images = [cv2.imread(p) for p in img_paths]
#
# # 计算颜色直方图统计特征
# features = []
# for img in images:
#     # 转换为HSV颜色空间
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # 计算HSV直方图
#     hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#
#     # 提取直方图统计特征
#     hist_mean = np.mean(hist)
#     hist_std = np.std(hist)
#     hist_skew = stats.skew(hist.flatten())
#     hist_kurt = stats.kurtosis(hist.flatten())
#
#     features.append([hist_mean, hist_std, hist_skew, hist_kurt])
#
# # 将特征转换为numpy数组
# features = np.array(features)
#
# # 输出特征统计信息
# print("Histogram Feature Statistics:")
# print("Mean:", features[:, 0].mean())
# print("Standard Deviation:", features[:, 1].mean())
# print("Skewness:", features[:, 2].mean())
# print("Kurtosis:", features[:, 3].mean())


import os
import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 设置图片路径
img_dir = './output/loose_s'

# 读取图像文件
img_files = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, f) for f in img_files]
images = [cv2.imread(p) for p in img_paths]

# 计算颜色直方图统计特征
features = []
for img in images:
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 计算HSV直方图
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # 提取直方图统计特征
    hist_mean = np.mean(hist)
    hist_std = np.std(hist)
    hist_skew = stats.skew(hist.flatten())
    hist_kurt = stats.kurtosis(hist.flatten())

    features.append([hist_mean, hist_std, hist_skew, hist_kurt])

# 将特征转换为numpy数组
features = np.array(features)

# 绘制特征的分布图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(features[:, 0], bins=20)
axes[0, 0].set_title('Histogram Mean')

axes[0, 1].hist(features[:, 1], bins=20)
axes[0, 1].set_title('Histogram Standard Deviation')

axes[1, 0].hist(features[:, 2], bins=20)
axes[1, 0].set_title('Histogram Skewness')

axes[1, 1].hist(features[:, 3], bins=20)
axes[1, 1].set_title('Histogram Kurtosis')

plt.tight_layout()
plt.show()