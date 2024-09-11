import os
import shutil

def synchronize_labels(images_folder, labels_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取images文件夹中的所有图片文件名（不包括后缀）
    image_names = set([os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith('.jpg')])
    # 获取labels文件夹中的所有标签文件名（不包括后缀）
    label_names = set([os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')])

    # 找出labels文件夹中多余的标签文件
    extra_label_names = label_names - image_names

    # 移动labels文件夹中多余的标签文件到输出文件夹
    for label_name in extra_label_names:
        label_path = os.path.join(labels_folder, label_name + '.txt')
        output_path = os.path.join(output_folder, label_name + '.txt')
        shutil.move(label_path, output_path)
        print(f"标签文件 {label_name}.txt 从 {labels_folder} 移动到 {output_folder}")

    print("标签文件同步完成！")

images_folder = r'./images'
labels_folder = r'./labels'
output_folder = r'./morelabels'
synchronize_labels(images_folder, labels_folder, output_folder)
