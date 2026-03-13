# -*- coding: utf-8 -*-
# ==============================================================================
# Multi-Model Classifier (多模型分类器) - Main Run Script
# 一站式运行脚本：自动切分数据集 -> 动态适配尺寸 -> 模型训练 -> 性能评估与成图
# 核心模型结构与训练代码分离，方便初学者学习和 DIY！
# ==============================================================================

import os
import sys
import shutil
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 获取当前文件所在目录及项目根目录 (Get current dir and root dir)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 如果 run.py 在根目录， ROOT_DIR = CURRENT_DIR
# 如果 run.py 在 src 目录， ROOT_DIR = os.path.dirname(CURRENT_DIR)
# 这里假设 run.py 在项目根目录，如果是在 src 下，使用 ROOT_DIR = os.path.dirname(CURRENT_DIR)
# 之前的 list_files 显示 run.py 在根目录 (GitHub/MVC-v5/MVC-v5/run.py) 
ROOT_DIR = CURRENT_DIR
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from Models import get_all_models
from Trainer import train_and_evaluate

# ==============================================================================
# 1. Global Configurations (全局参数配置 - 可以在这里修改所有参数！)
# ==============================================================================

# 模型选择 (Model Selection) - 在此配置你想要训练的模型，注释掉不想训练的即可
MODELS_TO_TRAIN = [
    # --- 经典网络 (Classic Networks) ---
    "C1_LeNet5",
    # "C2_AlexNet",
    # "C3_VGG16",
    # "C4_VGG19",
    "C5_ResNet18",
    # "C6_ResNet50",
    # "C7_DenseNet121",
    # "C8_MobileNetV2",

    # --- 自定义/手写网络 (Custom/DIY Networks) ---
    "C9_SimpleCNN",
    # "C10_MlpMixer",
    # "C11_GhostNet",
    # "C12_DepthwiseCNN",
    # "C13_ConvMixer"
]

# Data Splitting Ratios (数据集切分比例)
TRAIN_RATIO = 0.7    # 训练集比例
VAL_RATIO = 0.15     # 验证集比例
TEST_RATIO = 0.15    # 测试集比例

# Directories (目录配置)
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'Data_Raw')  # 用户放置原始图片的文件夹
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'Data') # 自动切分后保存的文件夹
OUTPUT_DIR = os.path.join(ROOT_DIR, 'OutPuts')      # 日志及结果图片保存文件夹

# Model configurations (模型配置)
IN_CHANNELS = 1      # 1: Grayscale (灰度图), 3: RGB (彩色图)
IMG_SIZE = 128       # 图像缩放尺寸 (例如: 128, 224)

# Training configurations (训练配置)
BATCH_SIZE = 128     # Batch size (批大小)
NUM_EPOCHS = 15      # Number of training epochs (训练轮数)
LEARNING_RATE = 1e-3 # Learning rate (学习率)

# Device configuration (设备配置)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 2. Dataset Split Utility (数据集自动切分工具)
# ==============================================================================
def split_dataset(raw_dir, processed_dir, train_ratio, val_ratio, test_ratio):
    """
    Automatically split images in Data_Raw into train, val, and test sets.
    自动将 Data_Raw 中的图片切分为训练集、验证集和测试集。
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0 (切分比例之和必须为1.0)"
    
    if not os.path.exists(raw_dir):
        print(f"[Info] {raw_dir} does not exist. Creating it... \n[提示] {raw_dir} 不存在，正在创建...")
        os.makedirs(raw_dir)
        print("Please put your images into class-folders inside 'Data_Raw', then re-run.\n请将图片按类别放入 'Data_Raw' 的子文件夹中，然后重新运行。")
        sys.exit(0)

    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if len(classes) < 2:
        print(f"\n[Warning] 🚨 警告：在 '{raw_dir}' 中仅检测到 {len(classes)} 个类别！\n普通的图像分类任务至少需要 2 个类别才能让模型学习到区分特征。\n如果只有一个类别，模型的准确率将永远是 100%，损失永远是 0，这是毫无意义的训练。\n由于您只有 '{classes[0]}' 这一个类别的文件夹，模型会自动把所有图片都猜成这个类别。\n\n程序将继续运行，但请注意上述情况！\n")
        
    print(f"[Info] Found classes (找到类别): {classes}")
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)
            
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        images = os.listdir(cls_dir)
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        random.shuffle(images)
        
        num_imgs = len(images)
        num_train = int(num_imgs * train_ratio)
        num_val = int(num_imgs * val_ratio)
        
        train_imgs = images[:num_train]
        val_imgs = images[num_train:num_train + num_val]
        test_imgs = images[num_train + num_val:]
        
        splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        
        for split, split_imgs in splits.items():
            for img in split_imgs:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(processed_dir, split, cls, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    
    print("[Info] Dataset splitting completed. (数据集切分完成。)\n")
    return len(classes)

# ==============================================================================
# 3. Main Execution (程序入口)
# ==============================================================================
if __name__ == '__main__':
    print(f"Using device (使用设备): {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Prepare Data (自动检测切分)
    print("\n--- Step 1: Checking and splitting dataset (检查并切分数据集) ---")
    num_classes_detected = split_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    NUM_CLASSES = num_classes_detected
    
    # 2. Data Transforms (自动适配通道配置)
    if IN_CHANNELS == 1:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    # 3. Load Data
    try:
        train_dataset = datasets.ImageFolder(root=os.path.join(PROCESSED_DATA_DIR, 'train'), transform=transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(PROCESSED_DATA_DIR, 'val'), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(PROCESSED_DATA_DIR, 'test'), transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        class_names = train_dataset.classes
        print(f"Classes (类别): {class_names}")
    except Exception as e:
        print(f"Data loading failed (数据加载失败): {e}")
        sys.exit(1)
        
    # 4. Define Models (获取独立文件中的模型结构)
    all_models_dict = get_all_models(IN_CHANNELS, NUM_CLASSES, IMG_SIZE)

    print("\n--- Step 2: Training Models (开始训练模型) ---")
    for name in MODELS_TO_TRAIN:
        if name in all_models_dict:
            model_instance = all_models_dict[name]
            train_and_evaluate(
                model_name=name, 
                model=model_instance, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                test_loader=test_loader, 
                num_epochs=NUM_EPOCHS, 
                learning_rate=LEARNING_RATE, 
                device=DEVICE,
                class_names=class_names,
                output_dir=OUTPUT_DIR
            )
        else:
            print(f"[Warning] Model '{name}' not found in Models.py! Skipping... (未找到模型，跳过...)")
