import os
import cv2
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.fsdecode = lambda s: s.decode('utf-8')

# 设置图像和标签文件夹路径
images_dir = './images'
labels_dir1 = './labels'
labels_dir2 = './Z1new'
output_dir = 'Z0-Real-Annotated'  # 输出文件夹

# 设置标注框的颜色和宽度
color1 = (0, 255, 0)  # Green for A0labels1-True
color2 = (0, 0, 255)  # Red for A1labels新增
thickness = 1

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(images_dir):
    if filename.endswith('.jpg'):
        # 读取图像
        image_path = os.path.join(images_dir, filename)
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Failed to read image: {image_path}")
            print(e)
            continue

        # 读取 A0labels1-True 标签文件
        label_path1 = os.path.join(labels_dir1, os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(label_path1):
            with open(label_path1, 'r') as f:
                labels1 = f.readlines()
        else:
            labels1 = []

        # 读取 A1labels新增 标签文件
        label_path2 = os.path.join(labels_dir2, os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(label_path2):
            with open(label_path2, 'r') as f:
                labels2 = f.readlines()
        else:
            labels2 = []

        # 遍历 A0labels1-True 标签并绘制绿色标注框
        for label in labels1:
            try:
                class_id, cx, cy, width, height = map(float, label.strip().split())
                class_id = int(class_id)

                # 计算标注框的坐标
                x = int((cx - width / 2) * image.shape[1])
                y = int((cy - height / 2) * image.shape[0])
                w = int(width * image.shape[1])
                h = int(height * image.shape[0])

                # 使用绿色绘制 A0labels1-True 标注框
                cv2.rectangle(image, (x, y), (x + w, y + h), color1, thickness)
            except ValueError:
                print(f"Invalid label format in file: {label_path1}")
                continue

        # 遍历 A1labels新增 标签并绘制红色标注框
        for label in labels2:
            try:
                class_id, cx, cy, width, height = map(float, label.strip().split())
                class_id = int(class_id)

                # 计算标注框的坐标
                x = int((cx - width / 2) * image.shape[1])
                y = int((cy - height / 2) * image.shape[0])
                w = int(width * image.shape[1])
                h = int(height * image.shape[0])

                # 使用红色绘制 A1labels新增 标注框
                cv2.rectangle(image, (x, y), (x + w, y + h), color2, thickness)
            except ValueError:
                print(f"Invalid label format in file: {label_path2}")
                continue

        # 保存标注后的图像
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image: {output_path}")