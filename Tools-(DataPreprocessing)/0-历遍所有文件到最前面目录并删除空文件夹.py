import os
import shutil

def get_unique_path(target_path):
    """
    如果目标路径已经存在，则在文件名后添加计数器，生成唯一路径。
    """
    base, extension = os.path.splitext(target_path)
    counter = 1
    unique_path = target_path

    while os.path.exists(unique_path):
        unique_path = f"{base}_{counter}{extension}"
        counter += 1

    return unique_path

# 指定要处理的目录
target_dir = "B6-All - 副本"

# 遍历所有子目录
for dirpath, dirnames, filenames in os.walk(target_dir):
    # 如果当前目录不是 target_dir 本身
    if dirpath != target_dir:
        # 将该目录下的所有文件移动到 target_dir
        for file in filenames:
            source_path = os.path.join(dirpath, file)
            target_path = os.path.join(target_dir, file)
            unique_target_path = get_unique_path(target_path)
            shutil.move(source_path, unique_target_path)
        # 删除当前空目录
        if not os.listdir(dirpath):
            os.rmdir(dirpath)

print("处理完成!")
