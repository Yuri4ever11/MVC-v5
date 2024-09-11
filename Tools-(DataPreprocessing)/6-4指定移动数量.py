import os
import random
import shutil

# 指定原始目录
source_dir = "/path/to/your/source/directory"

# 指定需要复制的文件数量
num_files_to_copy = 10

# 指定新目录的路径
new_dir = "/path/to/your/new/directory"

# 如果新目录不存在,则创建
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# 获取原始目录下的所有文件列表
all_files = os.listdir(source_dir)

# 从文件列表中随机选取指定数量的文件
selected_files = random.sample(all_files, num_files_to_copy)

# 将选定的文件复制到新目录
for file in selected_files:
    source_path = os.path.join(source_dir, file)
    target_path = os.path.join(new_dir, file)
    shutil.copy2(source_path, target_path)

print(f"已将 {num_files_to_copy} 个文件从 {source_dir} 复制到 {new_dir}.")