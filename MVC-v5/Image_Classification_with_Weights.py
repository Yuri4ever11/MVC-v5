import os
import shutil
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from tqdm import tqdm

# 加载模型！！！！！！不同模型需要进行单独修改！！！！！！！———————————————————————————————————————
model = models.resnet18(pretrained=False)
# 修改第一层卷积的输入通道数！！！！！！！重要！！！！！！—————————————————————————————————————————
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 修改为与加载的参数相同的形状
# 加载保存的模型参数-选择你需要的训练好的权重文件
model.load_state_dict(torch.load('./Weights/C5_ResNet18_LastEpoch.pth'))


# 设置模型为评估模式
model.eval()

# 数据预处理-可以自行设置-但是修改后需要同步修改 ↑ 内容
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 输入目录—————————————将你需要识别的类别放入，注意要放文件夹里面！！！———————————————————————————————————————
# 不要直接将图片放入，也不要一次放多个类别的文件夹！！！！
input_dir = './CategorizedImages/input'

# 输出目录————————————与你类别同名的文件夹里面存放被正确识别的图片！！！未被认出的则在unrecognized文件夹中—————————————
output_dir = './CategorizedImages/output'
# 输出目录中的另一个文件夹用于存放未识别的图片
unrecognized_dir = os.path.join(output_dir, 'unrecognized')
os.makedirs(unrecognized_dir, exist_ok=True)

# 置信度阈值！！！！！可以比较有效剔除难分辨的图片
confidence_threshold = 0.997

# 遍历输入目录下的所有文件夹
for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)
    if os.path.isdir(folder_path):
        # 创建对应的输出文件夹
        output_folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        # 使用 tqdm 进行进度条显示
        file_list = os.listdir(folder_path)
        progress_bar = tqdm(file_list, desc=f"Processing {folder_name}", leave=False)

        # 遍历文件夹内的图像文件
        for filename in progress_bar:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # 加载并预处理图像
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                image = transform(image).unsqueeze(0)  # 添加 batch 维度

                # 使用模型进行预测
                with torch.no_grad():
                    output = model(image)
                    prediction = torch.sigmoid(output)
                    max_prob, max_idx = torch.max(prediction, dim=1)
                    #——————————————————————————重要！下方数字代表模型需要判别的类别———————————————————————————————————
                    # 如果预测为类别索引为2,且置信度大于等于0.8,则将图像复制到输出文件夹中
                    if max_idx.item() == 2 and max_prob.item() >= confidence_threshold:
                        output_image_path = os.path.join(output_folder_path, filename)
                        shutil.copyfile(image_path, output_image_path)
                    else:
                        # 如果未识别为需要的类别,或置信度不足,则将图像复制到未识别的文件夹中
                        unrecognized_image_path = os.path.join(unrecognized_dir, filename)
                        shutil.copyfile(image_path, unrecognized_image_path)
                    # 更新进度条
                    progress_bar.set_postfix({'Processed': filename})

        # 关闭 tqdm 进度条
        progress_bar.close()
