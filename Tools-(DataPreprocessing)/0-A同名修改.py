import os

# 设置图像和标签文件夹路径
images_dir = 'Trainimages'
labels_dir = 'Trainlabels'

# 获取文件夹中的文件列表
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

# 检查文件数量是否相等
if len(image_files) != len(label_files):
    print("Number of image and label files do not match.")
    exit()

# 修改文件名
for i, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
    new_image_name = f"Train{i:04d}.jpg"
    new_label_name = f"Train{i:04d}.txt"

    # 修改图像文件名
    old_image_path = os.path.join(images_dir, image_file)
    new_image_path = os.path.join(images_dir, new_image_name)
    os.rename(old_image_path, new_image_path)
    print(f"Renamed image file: {image_file} -> {new_image_name}")

    # 修改标签文件名
    old_label_path = os.path.join(labels_dir, label_file)
    new_label_path = os.path.join(labels_dir, new_label_name)
    os.rename(old_label_path, new_label_path)
    print(f"Renamed label file: {label_file} -> {new_label_name}")

print("File renaming completed.")