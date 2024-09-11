import os
import shutil

# 设置文件夹路径
labels_folder = './labels/multi_class'
images_folder = './images/multi_class'
output_folders = ['class_0']

# 检查输出文件夹是否存在,如果不存在则创建
for folder in output_folders:
    if not os.path.exists(os.path.join(labels_folder, folder)):
        os.makedirs(os.path.join(labels_folder, folder))
    if not os.path.exists(os.path.join(images_folder, folder)):
        os.makedirs(os.path.join(images_folder, folder))

# 获取标签文件夹下所有txt文件
label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

# 遍历每个标签文件
for label_file in label_files:
    # 获取标签文件路径
    label_path = os.path.join(labels_folder, label_file)

    # 获取对应的图像文件路径
    image_name = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(images_folder, image_name)

    # 打开标签文件并读取内容
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except:
        print(f"Error reading {label_file}. Skipping...")
        continue

    # 检查是否包含类别 1
    has_class_1 = any(int(line.split()[0]) == 0 for line in lines)

    # 根据类别信息将标签文件和图像文件移动到对应的输出文件夹
    if has_class_1:
        dst_folder = os.path.join(labels_folder, 'class_0')
        dst_label_path = os.path.join(dst_folder, os.path.basename(label_path))
        dst_image_path = os.path.join(os.path.join(images_folder, 'class_0'), os.path.basename(image_path))
        if not os.path.exists(dst_label_path):
            shutil.move(label_path, dst_folder)
        else:
            print(f"Skipping {label_file} as the destination file already exists.")
        if not os.path.exists(dst_image_path):
            shutil.move(image_path, os.path.join(images_folder, 'class_0'))
        else:
            print(f"Skipping {image_name} as the destination file already exists.")