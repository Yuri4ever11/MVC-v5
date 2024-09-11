import os
from PIL import Image

# 标签映射关系
label_mapping = {
    0: "loose_l",
    1: "loose_s",
    2: "poor_l",
    3: "water_l"
}

def read_label_file(file_path):
    """
    读取标签文件，返回标签信息列表
    """
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            info = line.strip().split()
            if len(info) >= 5:
                labels.append({
                    'category': int(info[0]),
                    'center_x': float(info[1]),
                    'center_y': float(info[2]),
                    'width': float(info[3]),
                    'height': float(info[4])
                })
    return labels

def crop_and_save_images(images_folder, labels_folder, output_root_folder):
    """
    根据标签信息裁剪图片并保存到对应的文件夹中
    """
    os.makedirs(output_root_folder, exist_ok=True)
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            labels = read_label_file(os.path.join(labels_folder, label_file))
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            image_path = os.path.join(images_folder, image_name)
            with Image.open(image_path) as img:
                image_width, image_height = img.size
                for label in labels:
                    category = label['category']
                    label_name = label_mapping.get(category, "unknown")
                    output_folder = os.path.join(output_root_folder, label_name)
                    os.makedirs(output_folder, exist_ok=True)

                    center_x = label['center_x']
                    center_y = label['center_y']
                    width = label['width']
                    height = label['height']
                    left = (center_x - width / 2) * image_width
                    top = (center_y - height / 2) * image_height
                    right = (center_x + width / 2) * image_width
                    bottom = (center_y + height / 2) * image_height
                    cropped_img = img.crop((left, top, right, bottom))
                    center_x_str = f"{center_x:.18f}".rstrip('0')
                    center_y_str = f"{center_y:.18f}".rstrip('0')
                    width_str = f"{width:.18f}".rstrip('0')
                    height_str = f"{height:.18f}".rstrip('0')
                    output_image_path = os.path.join(output_folder, f"{os.path.splitext(label_file)[0]}-{category}-{center_x_str}-{center_y_str}-{width_str}-{height_str}.jpg")
                    cropped_img.save(output_image_path)

def main():
    images_folder = "./images"
    labels_folder = "./labels"
    output_root_folder = "./output"
    crop_and_save_images(images_folder, labels_folder, output_root_folder)
    print("图片分类并裁剪完成！")

if __name__ == "__main__":
    main()