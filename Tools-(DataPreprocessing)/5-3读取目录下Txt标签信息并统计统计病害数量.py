import os

# 设置文件夹路径
# labels_folder = 'H:\Z_AllData\A0-DataALL\DataA\SSIM0.4\labels\multi_class\class_1'
labels_folder = 'I:\THD\A0-T1\output提纯N_Labels'
# labels_folder = './multi_class/class_0'

# 初始化类别计数
category_counts = {0: 0, 1: 0, 2: 0, 3: 0}

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
    for line in lines:
        # 将每行内容分割成类别和坐标信息
        label_info = line.split()
        if len(label_info) < 1:
            print(f"Invalid line in {label_file}. Skipping...")
            continue
        category = int(label_info[0])

        # 更新类别计数
        if category in [0, 1, 2, 3]:
            category_counts[category] += 1

# 打印类别计数结果
print("Category Counts:")
for category, count in category_counts.items():
    print(f"Category {category}: {count}")