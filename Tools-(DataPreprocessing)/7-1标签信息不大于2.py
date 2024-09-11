import os
from shutil import copyfile

# 设置文件夹路径
labels1_dir = 'labels'
labels_dir = 'output提纯N_Labels'
new_dir = 'output提纯new_folder4'

# 创建新的文件夹
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# 遍历 labels1 文件夹中的所有文件
for filename in os.listdir(labels1_dir):
    # 构建文件路径
    labels1_file = os.path.join(labels1_dir, filename)
    labels_file = os.path.join(labels_dir, filename)

    # 检查文件是否存在
    if os.path.isfile(labels1_file) and os.path.isfile(labels_file):
        # 检查 labels1_file 中的行数是否不超过 labels_file 中行数的两倍
        with open(labels1_file, 'r') as f1, open(labels_file, 'r') as f2:
            lines1 = len(f1.readlines())
            lines2 = len(f2.readlines())
            if lines1 <= 2 * lines2:
                copyfile(labels1_file, os.path.join(new_dir, filename))
    else:
        print(f"Skipping {filename} as one of the files does not exist.")