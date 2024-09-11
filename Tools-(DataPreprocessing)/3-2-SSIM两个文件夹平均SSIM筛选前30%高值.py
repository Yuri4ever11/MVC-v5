import os
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
from collections import defaultdict
from heapq import nlargest


# 定义两个文件夹的路径
folder1 = './loose_l标准'
folder2 = './loose_l'
new_folder = './highSSIM'

# 创建新文件夹
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 存储每个图片的平均 SSIM 值
avg_ssim_dict = defaultdict(list)

# 遍历第二个文件夹中的所有图片
for filename2 in os.listdir(folder2):
    if filename2.endswith('.jpg') or filename2.endswith('.png'):
        img2 = imread(os.path.join(folder2, filename2)).astype(np.float32)

        # 计算第二个图片与第一个文件夹中所有图片的SSIM
        ssim_values = []
        for filename in os.listdir(folder1):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img1 = imread(os.path.join(folder1, filename)).astype(np.float32)

                # 调整图像尺寸
                img1_resized = resize(img1, img2.shape[:2])

                # 计算 SSIM
                ssim_value = ssim(img1_resized, img2, channel_axis=-1)
                ssim_values.append(ssim_value)

        # 计算平均SSIM
        avg_ssim = sum(ssim_values) / len(ssim_values)
        avg_ssim_dict[filename2].append(round(avg_ssim, 5))

        # 打印平均SSIM
        print(f"Average SSIM for {filename2} is {avg_ssim:.5f}")

# 对平均 SSIM 值进行排序
sorted_avg_ssim = sorted(avg_ssim_dict.items(), key=lambda x: x[1], reverse=True)

# 移动前 30% 的图片到新的文件夹
top_count = int(len(sorted_avg_ssim) * 0.4)
for i, (filename, ssim_values) in enumerate(sorted_avg_ssim):
    if i < top_count:
        src_path = os.path.join(folder2, filename)
        dst_path = os.path.join(new_folder, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved {filename} to {new_folder} with average SSIM {ssim_values[0]:.5f}")
    else:
        break