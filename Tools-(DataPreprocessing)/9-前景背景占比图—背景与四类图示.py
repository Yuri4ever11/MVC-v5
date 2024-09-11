import os
import matplotlib.pyplot as plt

# 定义标签类别
label_classes = ['loose_l', 'loose_s', 'poor_l', 'water_l']

# 定义函数来计算标签框面积
def calculate_bbox_area(labels_dir):
    total_bbox_areas = {label: 0 for label in label_classes}
    total_images = 0
    # 遍历标签文件夹
    for filename in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, filename), 'r') as file:
            lines = file.readlines()
            for line in lines:
                # 解析每行标签信息
                parts = line.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    label = label_classes[int(class_id)]
                    total_bbox_areas[label] += width * height
            total_images += 1

    return total_bbox_areas, total_images

# 定义函数来计算背景和其他四类的占比
def calculate_percentage(labels_dir):
    total_bbox_areas, total_images = calculate_bbox_area(labels_dir)

    # 计算总图片面积
    # 在此你需要根据你的数据集提供图片的总数量，因为标签文件不包含图片总数信息
    # 如果标签文件数量与图片数量一致，则可以直接使用标签文件数量来代替
    # 如果不一致，则需要提供图片总数
    # 简单起见，我们在此假设标签文件数量与图片数量一致
    total_images_area = total_images

    # 计算每类的占比
    percentages = {}
    for label, area in total_bbox_areas.items():
        percentage = area / total_images_area
        percentages[label] = percentage

    # 计算背景占比
    background_percentage = 1 - sum(percentages.values())

    return background_percentage, percentages

# 指定标签文件夹路径
labels_dir = r'F:\P_B_ObjectDetection\Yolov5-7.0\yolov5-7.0\mydata\labels\train'

# 计算背景和其他四类的占比
background_percentage, class_percentages = calculate_percentage(labels_dir)

# 输出结果
print("背景占比:", background_percentage)
for label, percentage in class_percentages.items():
    print(f"{label} 占比:", percentage)
# 定义函数来绘制柱状图
def plot_percentage(background_percentage, class_percentages):
    # 提取标签类别和对应的占比数据
    labels = list(class_percentages.keys())
    values = list(class_percentages.values())

    # 添加背景占比到数据中
    labels.append('Background')
    values.append(background_percentage)

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xlabel('类别')
    plt.ylabel('占比')
    plt.title('背景和四类的占比')
    plt.show()

# 调用函数绘制柱状图
plot_percentage(background_percentage, class_percentages)