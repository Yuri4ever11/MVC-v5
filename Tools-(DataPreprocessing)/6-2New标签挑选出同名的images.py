import os
import shutil

# 标签映射关系
label_mapping = {
    "loose_l": 0,
    "loose_s": 1,
    "poor_l": 2,
    "water_l": 3
}

def sync_yolo_dataset(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

    image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith(".txt")]

    # 确保每个图片都有对应的标签文件
    image_files_set = set([os.path.splitext(f)[0] for f in image_files])
    label_files_set = set([os.path.splitext(f)[0] for f in label_files])
    common_files = image_files_set.intersection(label_files_set)

    print(f"共有 {len(common_files)} 个文件匹配.")

    # 将图片和标签文件复制到输出文件夹
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        if image_name in common_files:
            shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, "images", image_file))

    for label_file in label_files:
        label_name = os.path.splitext(label_file)[0]
        if label_name in common_files:
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder, "labels", label_file))

    # 删除 images 文件夹中多余的文件
    for image_file in os.listdir(os.path.join(output_folder, "images")):
        image_name = os.path.splitext(image_file)[0]
        if image_name not in common_files:
            os.remove(os.path.join(output_folder, "images", image_file))

def main():
    images_folder = "./images"
    labels_folder = "./labels"
    output_folder = "./mydata"
    sync_yolo_dataset(images_folder, labels_folder, output_folder)
    print("YoloV5数据集同步完成!")

if __name__ == "__main__":
    main()