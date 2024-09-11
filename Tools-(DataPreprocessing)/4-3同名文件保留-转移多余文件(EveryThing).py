import os
import shutil

# 设置输入和输出文件夹路径
input_folder1 = 'SSIM0.5'
input_folder2 = 'images2'
output_folder = 'more'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取两个输入文件夹中的所有文件名
files1 = set(os.listdir(input_folder1))
files2 = set(os.listdir(input_folder2))

# 找出在 input_folder1 中与 input_folder2 中同名的文件
common_files = files1 & files2

# 移动在 input_folder1 中与 input_folder2 中同名的文件到输出文件夹
for file in common_files:
    file_path1 = os.path.join(input_folder1, file)
    output_file_path = os.path.join(output_folder, file)

    if os.path.isfile(file_path1):
        shutil.move(file_path1, output_file_path)
        print(f"Moved {file} from {input_folder1} to {output_folder}")
    else:
        print(f"Skipped {file} as it was not found in {input_folder1}")