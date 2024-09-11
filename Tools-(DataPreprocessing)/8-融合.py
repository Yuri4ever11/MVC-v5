import os

# 设置两个文件夹的路径
labels_folder = "labels5"
z1newl_folder = "Z1new"

# 创建融合后的文件夹
merged_folder = "merged"
os.makedirs(merged_folder, exist_ok=True)

# 遍历 labels 文件夹中的文件
for filename in os.listdir(labels_folder):
    if filename.endswith(".txt"):
        # 检查 Z1NewL 文件夹中是否有同名的文件
        if os.path.exists(os.path.join(z1newl_folder, filename)):
            # 读取两个文件夹中同名文件的内容
            with open(os.path.join(labels_folder, filename), "r") as f1, \
                    open(os.path.join(z1newl_folder, filename), "r") as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()

            # 将两个文件的内容进行融合,并去除空白行
            merged_lines = [line.strip() for line in lines1 + lines2 if line.strip()]

            # 将融合后的内容写入新的 txt 文件
            merged_path = os.path.join(merged_folder, filename)
            with open(merged_path, "w") as f:
                f.write("\n".join(merged_lines))

            print(f"Merged file: {filename}")
        else:
            # 如果 Z1NewL 文件夹中没有同名文件,则直接复制 labels 文件夹中的文件
            copied_path = os.path.join(merged_folder, filename)
            os.makedirs(os.path.dirname(copied_path), exist_ok=True)
            with open(os.path.join(labels_folder, filename), "r") as f1, \
                    open(copied_path, "w") as f2:
                f2.write(f1.read())

            print(f"Copied file: {filename}")