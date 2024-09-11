import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# 输入XML标签文件夹和输出YOLOv5标签文件夹
input_folder = r'A2-XML'
output_folder = r'A2-TXT'

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 创建类别映射字典
class_mapping = {
    "loose_l": 0,
    "loose_s": 1,
    "poor_l": 2,
    "water_l": 3
}

# 遍历XML标签文件夹中的每个XML文件
for xml_file in os.listdir(input_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(input_folder, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像的宽度和高度（YOLOv5使用归一化坐标）
            width = float(root.find('size').find('width').text)
            height = float(root.find('size').find('height').text)

            # 创建对应的YOLOv5格式的TXT文件
            txt_file = os.path.splitext(xml_file)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_file)

            # 打开TXT文件以写入坐标信息
            with open(txt_path, 'w') as txt_writer:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in class_mapping:
                        x_min = float(obj.find('bndbox').find('xmin').text)
                        y_min = float(obj.find('bndbox').find('ymin').text)
                        x_max = float(obj.find('bndbox').find('xmax').text)
                        y_max = float(obj.find('bndbox').find('ymax').text)

                        # 转换类别名称为数字ID
                        class_id = class_mapping[class_name]

                        # 计算YOLOv5格式的坐标（归一化坐标）
                        x_center = (x_min + x_max) / (2.0 * width)
                        y_center = (y_min + y_max) / (2.0 * height)
                        width_yolo = (x_max - x_min) / width
                        height_yolo = (y_max - y_min) / height

                        # 将坐标信息写入TXT文件
                        txt_writer.write(f"{class_id} {x_center} {y_center} {width_yolo} {height_yolo}\n")
        except ET.ParseError as e:
            print(f"Error parsing XML file: {xml_file}")
            print(e)

print("转换完成！")