# AIGO-Classify (多模型分类器)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![License](https://img.shields.io/badge/License-MIT-green)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Introduction
Welcome to the **AIGO-Classify** project! This project provides a modular, beautiful, and highly customizable framework for image classification tasks. It balances beginner-friendly automation with advanced DIY code structures.

### Key Features
- **Auto Data Splitting (`Data_Raw`)**: Just put your raw class-folders into `Data_Raw`. The script automatically shuffles and splits them into train/val/test sets inside the `Data` folder!
- **13+ Built-In & Handwritten Models**: Select exactly which models you want to train! We provide 13+ architectures out of the box, including custom handwritten models like `SimpleCNN`, `MlpMixer`, `GhostNet`, and `DepthwiseCNN`.
- **Modular Code Structure for Learning & DIY**:
  - `run.py`: The main entry point. All configurable parameters (epochs, batch size, model selection, etc.) are at the top. Keeps your execution simple!
  - `src/Models.py`: A dedicated file holding all 13+ network architectures. Perfectly clean for you to learn and manually DIY!
  - `src/Trainer.py`: Contains the deep learning training loop, validation, and testing logic.
  - `src/Plotter.py`: A beautifully optimized script to generate comprehensive plots (accuracy curves, confusion matrices, F1 scores) for each model.
- **Dynamic Model Input Size**: Automatically configures models based on your desired input size and channels.
- **Bilingual Comments**: The entire codebase contains smooth, easy-to-understand comments in both English and Chinese.

### Installation & Environment Setup
It is highly recommended to use **Python 3.10** with a virtual environment (like Conda).

1. **Create and activate a Conda environment** (Optional but recommended):
   ```bash
   conda create -n mvc_env python=3.10
   conda activate mvc_env
   ```
2. **Install the required packages**:
   We have prepared a modern `requirements.txt` file that ensures compatibility with newer Python versions and PyTorch 2.0+.
   ```bash
   pip install -r requirements.txt
   ```
   *(If you want to install GPU-accelerated PyTorch, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command tailored to your CUDA version).*

### How to Use
1. Place your raw image folders (named by class) into the `Data_Raw/` directory.
   ```
   Data_Raw/
   ├── cats/
   ├── dogs/
   └── birds/
   ```
2. Open `run.py` and modify your parameters at the top of the file:
   - Comment/Uncomment the models you want to train in the `MODELS_TO_TRAIN` list.
   - Adjust `TRAIN_RATIO`, `IMG_SIZE`, `IN_CHANNELS`, etc.
3. Run the script!
```bash
python run.py
```

---

<a name="chinese"></a>
## 中文

### 项目简介
欢迎使用 **AIGO-Classify**！本项目为您提供了一个模块化、高度可定制的深度学习图像分类框架。它不仅拥有傻瓜式的自动化运行能力，还保留了完整的独立模型代码，极其适合初学者学习与二次开发（DIY）！

### 核心特性
- **自动数据集切分**：只需将您的原始类别文件夹放入 `Data_Raw` 目录中。代码会自动帮您打乱并划分训练集、验证集和测试集，并放入 `Data` 文件夹中。
- **13+ 经典与手写模型**：支持在 `run.py` 中随心所欲地勾选想要训练的模型！内置了 LeNet, ResNet 等经典模型，以及 7 个自定义/手写模型（如简易版 MlpMixer, GhostNet, 深度可分离卷积等）。
- **专为学习与 DIY 设计的模块化结构**：
  - `run.py`：主运行脚本，极简设计！所有的参数设置、模型控制开关均整合在顶部，一目了然。
  - `src/Models.py`：独立的模型结构文件。支持动态尺寸，代码纯净，非常方便您手动修改和学习各种网络结构！
  - `src/Trainer.py`：独立的训练逻辑文件，包含训练、验证和测试循环。
  - `src/Plotter.py`：美观的成图代码，一键生成带有各类别精度曲线、混淆矩阵、F1分数的综合结果大图。
- **动态尺寸适配**：自动化设置模型的输入尺寸（例如 128x128、224x224）及通道数。
- **双语注释**：代码库全面升级了中英双语注释，语句更加通顺，方便不同语言背景的开发者阅读和学习。

### 环境安装指南
强烈建议使用 **Python 3.10**，并配合虚拟环境（如 Conda）进行部署。

1. **创建并激活 Conda 虚拟环境**（推荐）：
   ```bash
   conda create -n mvc_env python=3.10
   conda activate mvc_env
   ```
2. **安装所需依赖**：
   我们提供了一个更新后的 `requirements.txt` 文件，完美适配较新的 Python 版本和 PyTorch 2.0+ 框架。
   ```bash
   pip install -r requirements.txt
   ```
   *(如果您希望安装带 GPU 加速的 PyTorch，请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 寻找对应您 CUDA 版本的安装命令).*

### 使用方法
1. 在 `Data_Raw/` 目录下放入您的原始图片文件夹（以类别命名）。
   ```
   Data_Raw/
   ├── 猫/
   ├── 狗/
   └── 鸟/
   ```
2. 打开 `run.py`，您可以直接在最顶部的全局配置区域修改参数：
   - 在 `MODELS_TO_TRAIN` 列表中注释/取消注释您想要训练的模型。
   - 自定义 `TRAIN_RATIO`, `IMG_SIZE`, `IN_CHANNELS` 等。
3. 运行主训练脚本！
```bash
python run.py
```
