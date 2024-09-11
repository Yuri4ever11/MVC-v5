import os
import shutil

def synchronize_images(images_folder, labels_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取images文件夹中的所有图片文件名（不包括后缀）
    image_names = set([os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith('.jpg')])
    # 获取labels文件夹中的所有标签文件名（不包括后缀）
    label_names = set([os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')])

    # 找出images文件夹中多余的图片文件
    extra_image_names = image_names - label_names

    # 移动images文件夹中多余的图片文件到输出文件夹
    for image_name in extra_image_names:
        image_path = os.path.join(images_folder, image_name + '.jpg')
        output_path = os.path.join(output_folder, image_name + '.jpg')
        shutil.move(image_path, output_path)
        print(f"图片文件 {image_name}.jpg 从 {images_folder} 移动到 {output_folder}")

    print("图片文件同步完成！")

images_folder = r'./p'
labels_folder = r'./test'
output_folder = r'./moreimages'

synchronize_images(images_folder, labels_folder, output_folder)