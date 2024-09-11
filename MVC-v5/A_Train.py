# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import os
current_dir = os.getcwd()
import time
start_time = time.time()
import logging
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR
import torchvision.datasets as torchvision_datasets
import sys
import logging
import codecs
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import cv2
import os
import shutil
#----------3通道-224x224，需要注释另一个---------------------
# from Models_3_224 import *
from Models_1_128 import *
#----------单通道-128x128，需要注释另一个---------------------


# 创建 logger 函数——————————————————————————————————————————————————————————————————————————————
def create_logger(model_name):
    log_file_path = os.path.join('OutPuts', 'Log', f'{model_name}_log.txt')
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    fileHandler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')

    simple_formatter = logging.Formatter('%(message)s')

    consoleHandler.setFormatter(simple_formatter)
    fileHandler.setFormatter(simple_formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

# 在循环外创建 logger
global_logger = create_logger("Global_Log")

#————————————————————————————————————————————————————————————————————————————————————————————————
# 创建模型列表和名称列表,在此选择你想要的模型进行训练，可以一个也可以多个。num_classes=4 控制分类类别。
# 若想添加新的模型可以在Model_3_224.py或者Model_1_128.py中进行添加，新人写代码不是很规范，如果吐槽还请口下留情。
models = [
    C1_LeNet5(num_classes=4),
    C2_AlexNet(num_classes=4),
    C3_VGG16(num_classes=4),
    C3_VGG19(num_classes=4),
    C4_GoogLeNetV1(num_classes=4),
    C5_ResNet18(num_classes=4),
    C5_ResNet50(num_classes=4),
    C5_ResNet101(num_classes=4),
    C6_ResNeXt(num_classes=4),
    C7_DenseNet121(num_classes=4),
    C7_DenseNet161(num_classes=4),
    C7_DenseNet201(num_classes=4),
    C8_MobileNetV2(num_classes=4),
    C8_MobileNetV3(num_classes=4),
    C9_EfficientNet(num_classes=4),
    C10_DPN(num_classes=4),
    C11_IGCNet(num_classes=4),
    C12_Darknet53(num_classes=4)
]
model_names = [
    "C1_LeNet",
    "C2_AlexNet",
    "C3_VGG16",
    "C3_VGG19",
    "C4_GoogLeNetV1",
    "C5_ResNet18",
    "C5_ResNet50",
    "C5_ResNet101",
    "C6_ResNeXt",
    "C7_DenseNet121",
    "C7_DenseNet161",
    "C7_DenseNet201",
    "C8_MobileNetV2",
    "C8_MobileNetV3",
    "C9_EfficientNet",
    "C10_DPN",
    "C11_IGCNet",
    "C12_Darknet53"
]

#————————————————————————————————————————————————————————————————————————————————————————————————
## 数据预处理-单通道-输入128x128 注意！！！使用时需要在代码开头注释掉 #from Models_3_224 import *
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
# 数据预处理-3通道-输入224x224 注意！！！使用时需要在代码开头注释掉 #from Models_1_128 import *
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.495, 0.495, 0.495), (0.276, 0.276, 0.276))
# ])


# 加载数据集—————使用时注意DataA结构-train-test-val—————————————————————————————————————————————————
# 注意设置Batch_size大小(显存小则量力而行)-深度学习CNN不应该是高高在上，而是可以服务于每一台设备。^_^
data_dir = os.path.join(current_dir, 'Data', 'DataA')

train_dataset = torchvision_datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = torchvision_datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_dataset = torchvision_datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化每个类别的精度列表
train_acc_history = [[] for _ in range(len(train_dataset.classes))]
val_acc_history = [[] for _ in range(len(val_dataset.classes))]
test_acc_history = [[] for _ in range(len(test_dataset.classes))]

average_train = []
average_val = []
average_test = []
train_loss_history = []
val_loss_history = []
test_loss_history = []

# 在循环外部初始化最佳验证精度
best_val_accuracy = 0.0
# 循环模型
for model_instance, model_name in zip(models, model_names):
    print(f"\nTraining {model_name}...\n")
    # 创建一个新的模型实例
    model = model_instance
    model.to(device)

    # 加载预训练权重————————————————可以用，但是分类任务我几乎没用过，需要自己调试一下对于不同模型——————————————
    # pretrained_weights_path = "PrePTH/googlenet-1378be20.pth"
    # pretrained_dict = torch.load(pretrained_weights_path)
    # pretrained_dict.pop('fc.weight')
    # pretrained_dict.pop('fc.bias')
    # model.load_state_dict(pretrained_dict, strict=False)

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 训练模型-合理设置 Epoch 数需要综合考虑模型大小、数据集大小、学习率、数据复杂度、模型架构和硬件资源
    # 并通过实验找到在验证集上表现最佳的 Epoch 数，即不出现过拟合情况，验证集损失不回升，且较长轮数
    num_epochs = 15

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 重头-损失函数设置-我设置了多种损失函数来优化模型的数据不平衡问题，更适合的损失函数对结果影响很大。
    # ———————————————————————————————————①传统损失函数-交叉熵损失函数————————————————————————————————————
    criterion = nn.CrossEntropyLoss()

    # —————————————————②Focal Loss alpha参数用于调整每个类别的权重 gamma大会更关注于难样本—————————————————
    # alpha = torch.tensor([7, 29, 5, 5], dtype=torch.float32)
    # criterion = FocalLoss(alpha=alpha, gamma=4)

    # ———————③OhemCELoss-threshold高会更关注困难样本-min_kept保留最小困难样本数-ignore_label忽略的标签—————
    # criterion = OhemCELoss(threshold=0.65, min_kept=30, ignore_label=-1)

    # ————————————————————————④WeightedCrossEntropyLoss对每个类别进行参数调整———————————————————————————
    # for inputs, labels in train_loader:
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)
    #     # 前向传播
    #     outputs = model(inputs)
    # weight = torch.tensor([3,6, 1, 1], device=outputs.device, dtype=torch.float)
    # criterion = WeightedCrossEntropyLoss(weight=weight)

    #——⑤ClassBalancedLoss-Beta小会降低样本多的权重/高则增加，1是平衡-Gamma大会更关注难样本但可能会影响泛化性———
    # criterion = ClassBalancedLoss(4, beta=0.8, gamma=4.0)

    # ⑥BalancedSoftmaxLoss-beta小会导致类别权重对最近的类别频率更加敏感.beta大则会使类别权重对历史类别频率的影响更大
    # num_classes = 4
    # criterion = BalancedSoftmaxLoss(num_classes)
    # criterion = EqualizationLoss(num_classes)

    # ————————————————————————————⑦LDAMLoss+计算损失时传递参数要去模型代码中去修改————————————————————————
    # criterion = LDAMLoss(num_classes, max_m=0.5)

    # ———————————————若想添加更多的学习率可以去Models_3_224.py/Models_1_128.py文件中去添加————————————————

    # ！！！！！！配合学习了的设置同样非常重要，与学习率策略的设置，这里默认是余弦退火！！！！！！———————————————
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    # 创建一个新的模型实例——即初始的损失函数，如果上面的修改后出现BUG使用不了，靠这个重新开始编写。
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)



    # 在每次循环前创建新的 logger，并传入不同的模型名称
    logger = create_logger(model_name)

    # 初始化每个类别的精度列表
    train_acc_history = [[] for _ in range(len(train_dataset.classes))]
    val_acc_history = [[] for _ in range(len(val_dataset.classes))]
    test_acc_history = [[] for _ in range(len(test_dataset.classes))]

    # 训练模型  设置模型Epoch数
    logger.info(f'Model: {model_name}')

    # 训练模型
    best_val_accuracy = 0.0
    start_time = time.time()
    for epoch in range(num_epochs):
        # 在每个周期开始前更新学习率
        # scheduler.step()
        logger.info(f"Epoch: {epoch + 1}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = {i: 0 for i in range(len(train_dataset.classes))}
        total_train = {i: 0 for i in range(len(train_dataset.classes))}

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            for c in range(len(train_dataset.classes)):
                class_indices = labels == c
                total_train[c] += class_indices.sum().item()
                correct_train[c] += (predicted[class_indices] == labels[class_indices]).sum().item()

        # 计算每个类别精度
        class_names = train_dataset.classes
        for c in range(len(train_dataset.classes)):
            if total_train[c] > 0:
                train_accuracy = 100 * correct_train[c] / total_train[c]
                train_acc_history[c].append(train_accuracy)
                logger.info(f'Training Accuracy for {class_names[c]}: {train_accuracy:.2f}%')
            else:
                logger.info(f'Training Accuracy for {class_names[c]}: No training samples for this class.')

        # 计算平均精度
        average_train_accuracy = np.mean(
            [train_acc_history[i][-1] for i in range(len(train_dataset.classes)) if train_acc_history[i]])
        logger.info(f'average_train_accuracy: {average_train_accuracy:.4f}')
        logger.info(f'Training loss : {train_loss:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = {i: 0 for i in range(len(val_dataset.classes))}
        total_val = {i: 0 for i in range(len(val_dataset.classes))}

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                for c in range(len(val_dataset.classes)):
                    class_indices = labels == c
                    total_val[c] += class_indices.sum().item()
                    correct_val[c] += (predicted[class_indices] == labels[class_indices]).sum().item()

            # 计算每个类别精度（验证集）
            class_names = val_dataset.classes
            for c in range(len(val_dataset.classes)):
                val_accuracy = 100 * correct_val[c] / total_val[c]
                val_acc_history[c].append(val_accuracy)
                logger.info(f'Validation Accuracy for {class_names[c]}: {val_accuracy:.2f}%')

            # 计算平均验证精度
            average_val_accuracy = np.mean([val_acc_history[i][-1] for i in range(len(train_dataset.classes))])
            logger.info(f'Average Validation Accuracy: {average_val_accuracy:.4f}')
            logger.info(f'Validation Loss: {val_loss:.4f}')

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if average_val_accuracy > best_val_accuracy:
            best_val_accuracy = average_val_accuracy

            # 保存模型
            model_dir = os.path.join('Weights')
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, f'{model_name}_BestEpoch.pth')
            torch.save(model.state_dict(), model_save_path)
            # 每一轮 epoch 都打印此信息
            # logger.info(f'Model saved at {model_save_path}')
            # 只在最后一个 epoch 打印此信息
            if epoch == num_epochs - 1:
                logger.info(f'Model saved at {model_save_path}')

        # 在最后一个 epoch 结束后进行测试
        if epoch == num_epochs - 1:
            logger.info("Testing model on the test set...")
            model.eval()
            test_loss = 0.0
            correct_test = {i: 0 for i in range(len(test_dataset.classes))}
            total_test = {i: 0 for i in range(len(test_dataset.classes))}

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    for c in range(len(test_dataset.classes)):
                        class_indices = labels == c
                        total_test[c] += class_indices.sum().item()
                        correct_test[c] += (predicted[class_indices] == labels[class_indices]).sum().item()

                # 计算每个类别精度
                class_names = test_dataset.classes
                for c in range(len(test_dataset.classes)):
                    test_accuracy = 100 * correct_test[c] / total_test[c]
                    test_acc_history[c].append(test_accuracy)
                    logger.info(f'Test Accuracy for {class_names[c]}: {test_accuracy:.2f}%')

                # 计算平均测试精度
                average_test_accuracy = np.mean([test_acc_history[i][-1] for i in range(len(test_dataset.classes))])
                logger.info(f'Average Test Accuracy: {average_test_accuracy:.4f}')
                logger.info(f'Test Loss: {test_loss:.4f}')

                # 保存最后一个 epoch 的模型
                model_dir = os.path.join('Weights')
                os.makedirs(model_dir, exist_ok=True)
                model_save_path = os.path.join(model_dir, f'{model_name}_LastEpoch.pth')
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Model saved at {model_save_path}')

                # 输出当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Current Learning Rate: {current_lr}')

                # 在每个 epoch 结束后执行一次的代码块
                if epoch == num_epochs - 1:  # 只在最后一个 epoch 结束后执行一次

                    # 生成混淆矩阵和 F1 分数
                    model.eval()
                    all_labels = []
                    all_predictions = []

                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            _, predicted = outputs.max(1)

                            all_labels.extend(labels.cpu().numpy())
                            all_predictions.extend(predicted.cpu().numpy())

                    # 计算混淆矩阵和 F1 分数
                    conf_matrix = confusion_matrix(all_labels, all_predictions)
                    f1_scores = f1_score(all_labels, all_predictions, average=None)
                    average_test_accuracy = np.mean([test_acc_history[i][-1] for i in range(len(test_dataset.classes))])

                    # 在每个 epoch 结束后确保历史列表长度为 num_epochs
                    for c in range(len(train_dataset.classes)):
                        if len(train_acc_history[c]) < num_epochs:
                            train_acc_history[c].extend(
                                [train_acc_history[c][-1]] * (num_epochs - len(train_acc_history[c])))

                    for c in range(len(val_dataset.classes)):
                        if len(val_acc_history[c]) < num_epochs:
                            val_acc_history[c].extend([val_acc_history[c][-1]] * (num_epochs - len(val_acc_history[c])))

                    for c in range(len(test_dataset.classes)):
                        if len(test_acc_history[c]) < num_epochs:
                            test_acc_history[c].extend(
                                [test_acc_history[c][-1]] * (num_epochs - len(test_acc_history[c])))

                    # 打印混淆矩阵和 F1 分数
                    print("\nConfusion Matrix:")
                    print(conf_matrix)
                    print("\nF1 Scores:")
                    print(f1_scores)

                    # 计算并打印平均测试精度
                    print(f"\nAverage Test Accuracy: {average_test_accuracy:.2f}%")

                    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————
                    """
            !!!注意一下，小生在写码之初只考虑了当时项目的四分类任务，故该模块只适应于4分类任务的成图设计。!!!
                    ！！不过也不用担心，该项目还有别的成图代码用于生成图，没有限制，可以随意DIY ！！       
                    """
                    # 图像生成设计——如果不是四分类可以 # 这一段 (鼠标选中按住左CTRL+/)
                    # 创建一个包含 len(class_names)//2 + 1 行两列的网格
                    fig, axes = plt.subplots(len(class_names) // 2 + 1, 2, figsize=(10, 15))

                    # 设置全局字体
                    plt.rcParams['font.family'] = 'Times New Roman'
                    plt.rcParams['font.size'] = 14  # 设置全局字体大小

                    # 绘制每个类别的训练、验证和测试精度曲线
                    for i in range(len(class_names)):
                        row = i // 2
                        col = i % 2
                        ax = axes[row, col]
                        # 使用平滑曲线绘制
                        ax.plot(range(1, len(train_acc_history[i]) + 1), train_acc_history[i], label='Train Accuracy',
                                linewidth=2, linestyle='-', alpha=0.7, marker='o', markersize=4)  # 添加圆形标记
                        ax.plot(range(1, len(val_acc_history[i]) + 1), val_acc_history[i], label='Validation Accuracy',
                                linewidth=2, linestyle='-', alpha=0.7, marker='o', markersize=4)  # 添加圆形标记
                        # 删除测试精度曲线
                        # ax.plot(range(1, len(test_acc_history[i]) + 1), test_acc_history[i], label='Test Accuracy')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Accuracy')
                        ax.set_title(
                            f'Class: {class_names[i]} \nTrain Acc: {train_acc_history[i][-1]:.2f}%\nVal Acc: {val_acc_history[i][-1]:.2f}%')
                        ax.set_ylim(0, 100)  # 设置纵坐标范围
                        ax.legend()

                    # 计算混淆矩阵和F1分数
                    conf_matrix = confusion_matrix(all_labels, all_predictions)
                    f1_scores = f1_score(all_labels, all_predictions, average=None)
                    average_test_accuracy = np.mean([test_acc_history[i][-1] for i in range(len(test_dataset.classes))])

                    # 在第三行第一列的子图中绘制混淆矩阵、F1分数和平均测试精度
                    confusion_text = f"Confusion Matrix:\n{conf_matrix}\n\nF1 Scores:\n{f1_scores}\n\nAverage Test Accuracy: {average_test_accuracy:.2f}%"
                    ax = axes[2, 0]
                    total_training_time = time.time() - start_time
                    total_training_minutes = total_training_time / 3600
                    confusion_text += f"\n\nTotal Training Time: {total_training_minutes:.2f} H"
                    ax.text(0.5, 0.85, confusion_text, wrap=True, horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12)
                    ax.axis('off')

                    # 调整第三行第二列的子图为空白
                    axes[2, 1].axis('off')

                    # 调整子图之间的间距和布局
                    fig.tight_layout()

                    # 保存方框图
                    plot_dir = os.path.join(current_dir, 'OutPuts')
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(plot_dir, f'{model_name}_1.png')
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Combined plot saved at: {plot_path}")



# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            """
            这一段考虑到特殊需求而创的可以识别被错误分类的图片，在第五个大版本也就是这个版本上似乎遇到了一些问题无法使用。
            如果使用的诸位有这个需求可以去使用上一个版本。
            还有别的方式就是使用该项目中的【单模型识别】直接判断是否可以正确识别某个类别。
            """
            # # 创建一个用于保存错误图片的文件夹
            # errors_dir = os.path.join(current_dir, './OutPuts/Errors')
            # os.makedirs(errors_dir, exist_ok=True)
            #
            # # 获取测试集的真实标签
            # all_labels = []
            # for i in range(len(test_dataset)):
            #     all_labels.append(test_dataset[i][1])  # 假设真实标签是 test_dataset[i][1]
            #
            # # 在测试集上进行预测
            # all_predictions = []
            # with torch.no_grad():
            #     for i in range(len(test_dataset)):
            #         input_data = test_dataset[i][0].unsqueeze(0).to(device)
            #         output = model(input_data)
            #         _, predicted = output.max(1)
            #         all_predictions.append(predicted.item())
            #
            # # 遍历所有样本，将预测错误的样本保存到新的文件夹中
            # for i in range(len(all_labels)):
            #     # 获取原始图像路径
            #     original_image_path = test_dataset.samples[i][0]
            #
            #     # 获取原始图像名称
            #     original_image_name = os.path.basename(original_image_path)
            #
            #     # 获取预测结果
            #     predicted_label = all_predictions[i]
            #
            #     # 获取真实标签
            #     true_label = all_labels[i]
            #
            #     # 构造保存图像的文件名
            #     image_name = f"{true_label}_{original_image_name}_{predicted_label}.png"
            #
            #     # 加载输入数据，确保与预测结果对应
            #     input_data = test_dataset[i][0].unsqueeze(0).to(device)
            #
            #     # 将图像数据从tensor转换为numpy数组，并确保是单通道的
            #     image_data = np.squeeze((input_data.cpu().numpy() * 255).astype(np.uint8))
            #
            #     # 判断预测结果是否正确
            #     if predicted_label != true_label:
            #         # 保存到错误分类文件夹
            #         save_path = os.path.join(errors_dir, image_name)
            #         cv2.imwrite(save_path, image_data)
# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————

# 关闭当前模型的 logger
logger.handlers.clear()

# 关闭全局 logger
global_logger.handlers.clear()