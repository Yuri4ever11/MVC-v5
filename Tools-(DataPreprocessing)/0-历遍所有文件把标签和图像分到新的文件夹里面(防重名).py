import os
import shutil
import re

# 设置源文件夹路径
source_label_folder = "标签"
source_image_folder = "图像"

# 设置目标文件夹路径
target_label_folder = "labels"
target_image_folder = "images"

# 创建目标文件夹
if not os.path.exists(target_label_folder):
    os.makedirs(target_label_folder)
if not os.path.exists(target_image_folder):
    os.makedirs(target_image_folder)

# 遍历标签文件夹及子文件夹
for root, dirs, files in os.walk(source_label_folder):
    for file in files:
        if file.endswith(".xml"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_label_folder, file)
            # 检查目标文件夹中是否已经存在同名文件
            if os.path.exists(dst_path):
                # 如果存在,则修改文件名
                base, ext = os.path.splitext(file)
                new_base = base + "_1"
                new_file = new_base + ext
                new_dst_path = os.path.join(target_label_folder, new_file)
                i = 2
                while os.path.exists(new_dst_path):
                    new_base = base + f"_{i}"
                    new_file = new_base + ext
                    new_dst_path = os.path.join(target_label_folder, new_file)
                    i += 1
                dst_path = new_dst_path
            shutil.move(src_path, dst_path)

# 遍历图像文件夹及子文件夹
for root, dirs, files in os.walk(source_image_folder):
    for file in files:
        if file.endswith(".jpg"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_image_folder, file)
            # 检查目标文件夹中是否已经存在同名文件
            if os.path.exists(dst_path):
                # 如果存在,则修改文件名
                base, ext = os.path.splitext(file)
                new_base = base + "_1"
                new_file = new_base + ext
                new_dst_path = os.path.join(target_image_folder, new_file)
                i = 2
                while os.path.exists(new_dst_path):
                    new_base = base + f"_{i}"
                    new_file = new_base + ext
                    new_dst_path = os.path.join(target_image_folder, new_file)
                    i += 1
                dst_path = new_dst_path
            shutil.move(src_path, dst_path)

print("文件移动完成!")