import os
import shutil

# 设置文件夹路径
labels_folder = './labels\multi_class'
output_folder = './labels\multi_class/class_0'

# 检查输出文件夹是否存在,如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取标签文件夹下所有 txt 文件
label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

# 遍历每个标签文件
for label_file in label_files:
    # 获取标签文件路径
    label_path = os.path.join(labels_folder, label_file)

    # 打开标签文件并读取内容
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except:
        print(f"Error reading {label_file}. Skipping...")
        continue

    # 检查是否包含类别 0
    has_class_0 = any(int(line.split()[0]) == 0 for line in lines)

    # 根据类别信息将标签文件移动到对应的输出文件夹
    if has_class_0:
        dst_label_path = os.path.join(output_folder, os.path.basename(label_path))
        if not os.path.exists(dst_label_path):
            shutil.move(label_path, output_folder)
        else:
            print(f"Skipping {label_file} as the destination file already exists.")