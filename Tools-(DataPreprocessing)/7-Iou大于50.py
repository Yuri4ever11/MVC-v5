import os
import numpy as np

# 设置文件路径
labels1_path = "labels/"
labels2_path = "labels-5s/"
images_path = "images/"
output_path = "Z1Newlabels/"

# 创建输出文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)


def get_iou(box1, box2):
    """
    计算两个标签框的IoU
    """
    x1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    y1 = max(box1[2] - box1[4] / 2, box2[2] - box2[4] / 2)
    x2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
    y2 = min(box1[2] + box1[4] / 2, box2[2] + box2[4] / 2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1[3] * box1[4] + box2[3] * box2[4] - intersection
    return intersection / union


def filter_overlapping_boxes(boxes):
    """
    过滤掉同一个区域中重叠的标签框,保留面积最大的那个。
    """
    # 按照面积从大到小排序
    boxes = sorted(boxes, key=lambda x: x[3] * x[4], reverse=True)

    # 遍历标签框,检查是否有重叠
    filtered_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[i]
        is_overlapping = False
        for j in range(len(filtered_boxes)):
            box_j = filtered_boxes[j]
            # 如果两个标签框的中心点距离小于等于标签框宽度和高度的平均值的50%,则认为它们重叠
            if abs(box_i[1] - box_j[1]) <= (box_i[3] + box_j[3]) / 8 and abs(box_i[2] - box_j[2]) <= (
                    box_i[4] + box_j[4]) / 8:
                is_overlapping = True
                # 保留面积更大的标签框
                if box_i[3] * box_i[4] > box_j[3] * box_j[4]:
                    filtered_boxes[j] = box_i
                break
        if not is_overlapping:
            filtered_boxes.append(box_i)

    return filtered_boxes


# 遍历labels2文件夹中的所有文件
for filename in os.listdir(labels2_path):
    if filename.endswith(".txt"):
        # 尝试读取labels1和labels2文件
        try:
            with open(os.path.join(labels1_path, filename), 'r') as f:
                labels1 = f.readlines()
        except FileNotFoundError:
            print(f"Warning: {filename} not found in labels1 folder. Skipping this file.")
            continue

        try:
            with open(os.path.join(labels2_path, filename), 'r') as f:
                labels2 = f.readlines()
        except FileNotFoundError:
            print(f"Warning: {filename} not found in labels2 folder. Skipping this file.")
            continue

        # 创建一个新的labels2列表,用于存储更新后的标签
        new_labels2 = []

        # 遍历labels2中的每个标签,先与labels1中的标签进行对比,再与自身进行对比
        labels2_boxes = []
        for label2 in labels2:
            label2_split = label2.strip().split()
            label2_class = int(label2_split[0])
            label2_box = np.array([label2_class, *[float(x) for x in label2_split[1:]]])

            # 与labels1中的标签框进行对比,如果IoU大于0.0001,则删除该标签框
            skip_label = False
            for label1 in labels1:
                label1_split = label1.strip().split()
                label1_box = np.array([int(label1_split[0]), *[float(x) for x in label1_split[1:]]])
                if get_iou(label2_box, label1_box) > 0.0001:
                    skip_label = True
                    break
            if skip_label:
                continue

            labels2_boxes.append(label2_box)

        # 过滤掉同一个区域中重叠的标签框
        filtered_labels2_boxes = filter_overlapping_boxes(labels2_boxes)

        # 将过滤后的标签框写入new_labels2列表
        for box in filtered_labels2_boxes:
            new_labels2.append(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}")

        # 将更新后的标签写入新的labels2文件
        with open(os.path.join(output_path, filename), 'w') as f:
            f.writelines('\n'.join(new_labels2))
        print(f"Processed file: {filename}")