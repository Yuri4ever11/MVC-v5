import os
import shutil
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

def filter_images_by_ssim(base_folder, target_folder, ssim_threshold=0.4):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取文件夹中的所有图片文件
    images = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 以第一张图片为基准
    base_img_path = images[0]
    base_img = img_as_float(io.imread(base_img_path, as_gray=True))
    base_data_range = base_img.max() - base_img.min()

    for img_path in images[1:]:
        img = img_as_float(io.imread(img_path, as_gray=True))
        img_resized = resize(img, base_img.shape, anti_aliasing=True)
        ssim_value, _ = ssim(base_img, img_resized, full=True, data_range=base_data_range)

        if ssim_value < ssim_threshold:
            shutil.move(img_path, os.path.join(target_folder, os.path.basename(img_path)))
            print(f"图片 {os.path.basename(img_path)} 已移动到 {target_folder}，SSIM值为: {ssim_value:.5f}")

base_folder = 'images'
target_folder = './SSIM0.4'

filter_images_by_ssim(base_folder, target_folder)