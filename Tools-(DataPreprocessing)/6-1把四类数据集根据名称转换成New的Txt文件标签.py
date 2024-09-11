import os

def generate_label_files(output_root_folder, labels_folder):
    """
    根据 output 文件夹中的图片信息,生成标签文件
    """
    os.makedirs(labels_folder, exist_ok=True)

    for label_dir in os.listdir(output_root_folder):
        label_dir_path = os.path.join(output_root_folder, label_dir)
        if os.path.isdir(label_dir_path):
            for image_file in os.listdir(label_dir_path):
                image_path = os.path.join(label_dir_path, image_file)
                parts = image_file.rsplit("-", 5)
                label_file_name = "-".join(parts[:-5])
                height = float(parts[-1].replace(".jpg", ""))
                width = float(parts[-2])
                center_y = float(parts[-3])
                center_x = float(parts[-4])
                category = int(float(parts[-5]))

                label_file_path = os.path.join(labels_folder, f"{label_file_name}.txt")
                with open(label_file_path, "a") as label_file:
                    label_file.write(f"{category} {center_x} {center_y} {width} {height}\n")

def main():
    output_root_folder = "./labels"
    labels_folder = "./labelsoutputNEW"
    generate_label_files(output_root_folder, labels_folder)
    print("标签文件生成完成!")

if __name__ == "__main__":
    main()