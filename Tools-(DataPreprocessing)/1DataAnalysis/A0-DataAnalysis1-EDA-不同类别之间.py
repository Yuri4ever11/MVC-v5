import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图片路径
img_dir = './output/loose_s'

# 读取图像文件
img_files = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, f) for f in img_files]
images = [plt.imread(p) for p in img_paths]

# 图像数据统计
num_images = len(images)
image_sizes = [img.shape for img in images]
image_channels = [img.shape[-1] for img in images if len(img.shape) > 2]

print(f"Number of images: {num_images}")
print(f"Image sizes: {image_sizes}")
print(f"Image channels: {image_channels}")

# 图像尺寸分布
plt.figure(figsize=(8, 6))
sns.histplot([s[0] for s in image_sizes], bins=20, kde=True)
plt.title("Image Height Distribution")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot([s[1] for s in image_sizes], bins=20, kde=True)
plt.title("Image Width Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
plt.show()

# 图像像素值分布
flattened_images = np.concatenate([img.flatten() for img in images])
plt.figure(figsize=(8, 6))
sns.histplot(flattened_images, bins=50, kde=True)
plt.title("Pixel Value Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Count")
plt.show()

# 图像相关性分析(仅适用于RGB图像)
if 3 in image_channels:
    corr_matrix = np.corrcoef([img.flatten() for img in images])
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap="YlOrRd", annot=True)
    plt.title("Pixel Correlation Heatmap")
    plt.show()