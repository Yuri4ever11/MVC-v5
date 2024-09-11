import os
import shutil

# 设置文件夹路径
labels_folder = 'labels'
output_folders = ['0', '1', '2', '3', 'multi_class']

# 检查输出文件夹是否存在,如果不存在则创建
for folder in output_folders:
    if not os.path.exists(os.path.join(labels_folder, folder)):
        os.makedirs(os.path.join(labels_folder, folder))

# 获取标签文件夹下所有txt文件
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

    # 遍历标签文件中的每行内容
    categories = set()
    for line in lines:
        # 将每行内容分割成类别和坐标信息
        label_info = line.split()
        if len(label_info) < 1:
            print(f"Invalid line in {label_file}. Skipping...")
            continue
        category = int(label_info[0])
        categories.add(category)

    # 根据类别信息将标签文件移动到对应的输出文件夹
    if len(categories) == 1:
        # 只有一类别的情况
        category = list(categories)[0]
        dst_folder = os.path.join(labels_folder, str(category))
        dst_label_path = os.path.join(dst_folder, os.path.basename(label_path))
        if not os.path.exists(dst_label_path):
            shutil.move(label_path, dst_folder)
        else:
            print(f"Skipping {label_file} as the destination file already exists.")
    else:
        # 有多类别的情况
        dst_folder = os.path.join(labels_folder, 'multi_class')
        dst_label_path = os.path.join(dst_folder, os.path.basename(label_path))
        if not os.path.exists(dst_label_path):
            shutil.move(label_path, dst_folder)
        else:
            print(f"Skipping {label_file} as the destination file already exists.")