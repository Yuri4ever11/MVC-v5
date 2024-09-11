import os
import shutil
import random

# 标签映射关系
label_mapping = {
    "loose_l": 0,
    "loose_s": 1,
    "poor_l": 2,
    "water_l": 3
}

# 每个类别的目标样本数量
target_sample_count_per_class = 5468

def prepare_yolo_dataset(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "val"), exist_ok=True)

    image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith(".txt")]

    # 确保每个图片都有对应的标签文件
    image_files_set = set([os.path.splitext(f)[0] for f in image_files])
    label_files_set = set([os.path.splitext(f)[0] for f in label_files])
    common_files = image_files_set.intersection(label_files_set)

    print(f"共有 {len(common_files)} 个文件匹配.")

    # 统计每个类别的样本数量
    category_counts = {}
    for label_file in label_files:
        label_name = os.path.splitext(label_file)[0]
        if label_name in common_files:
            with open(os.path.join(labels_folder, label_file), "r") as f:
                category = int(f.read().strip().split()[0])
                category_name = list(label_mapping.keys())[list(label_mapping.values()).index(category)]
                if category_name not in category_counts:
                    category_counts[category_name] = 0
                category_counts[category_name] += 1

    # 将图片和标签文件复制到训练集和验证集
    train_ratio = 0.8
    for image_file, label_file in zip(image_files, label_files):
        image_name = os.path.splitext(image_file)[0]
        label_name = os.path.splitext(label_file)[0]
        if image_name in common_files and label_name in common_files:
            with open(os.path.join(labels_folder, label_file), "r") as f:
                category = int(f.read().strip().split()[0])
                category_name = list(label_mapping.keys())[list(label_mapping.values()).index(category)]

            if category_counts[category_name] < target_sample_count_per_class:
                if random.random() < train_ratio:
                    shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, "images", "train", image_file))
                    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder, "labels", "train", label_file))
                else:
                    shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, "images", "val", image_file))
                    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder, "labels", "val", label_file))
                category_counts[category_name] += 1
            elif category_counts[category_name] == target_sample_count_per_class:
                if random.random() < train_ratio:
                    shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, "images", "train", image_file))
                    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder, "labels", "train", label_file))
                else:
                    shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, "images", "val", image_file))
                    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder, "labels", "val", label_file))

def main():
    images_folder = "./images"
    labels_folder = "./labels"
    output_folder = "./yolo_dataset"
    prepare_yolo_dataset(images_folder, labels_folder, output_folder)
    print("YoloV5数据集准备完成!")

if __name__ == "__main__":
    main()