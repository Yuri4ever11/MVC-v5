import os
import re
import matplotlib.pyplot as plt




"""
这个直接运行就好 有txt训练即可，自由度很高，可以各种DIY。
Loss图非常的重要，需要重点去关注，生成的图片会出现在Outputs/Plots中。
"""
def extract_info(log_path):
    with open(log_path, 'r') as file:
        content = file.read()

    # 使用正则表达式提取信息
    epochs = re.findall(r'Epoch: (\d+)', content)

    train_accuracies = {}
    val_accuracies = {}
    # test_accuracies = {}

    for match in re.finditer(r'Training Accuracy for (\w+): ([\d.]+)%', content):
        label, acc = match.group(1), match.group(2)
        train_accuracies.setdefault(label, []).append(float(acc))

    avg_train_accuracy_match = re.search(r'average_train_accuracy: (\d+\.\d+)', content)
    avg_train_accuracy = avg_train_accuracy_match.group(1) if avg_train_accuracy_match else None

    for match in re.finditer(r'Validation Accuracy for (\w+): ([\d.]+)%', content):
        label, acc = match.group(1), match.group(2)
        val_accuracies.setdefault(label, []).append(float(acc))

    avg_val_accuracy_match = re.search(r'Average Validation Accuracy: (\d+\.\d+)', content)
    avg_val_accuracy = avg_val_accuracy_match.group(1) if avg_val_accuracy_match else None

    # 提取损失值并转换为列表
    train_loss = [float(match.group(1)) for match in re.finditer(r'Training loss : (\d+\.\d+)', content)]
    val_loss = [float(match.group(1)) for match in re.finditer(r'Validation Loss: (\d+\.\d+)', content)]

    return epochs, train_accuracies, avg_train_accuracy, val_accuracies, avg_val_accuracy, train_loss, val_loss


# 生成子图并保存
def generate_plots(log_path):
    epochs, train_accuracies, avg_train_accuracy, val_accuracies, avg_val_accuracy, train_loss, val_loss = extract_info(log_path)

    # 转换数据类型
    epochs = list(map(int, epochs))

    # 创建子图
    fig, axs = plt.subplots(3, 1, figsize=(18, 25))  # 3 行 1 列

    # 训练集精度和平均精度
    for label, acc_values in train_accuracies.items():
        axs[0].plot(epochs[:len(acc_values)], acc_values, label=label)

    avg_train_values = [sum(values) / len(values) for values in zip(*train_accuracies.values())]
    axs[0].plot(epochs[:len(avg_train_values)], avg_train_values, marker='o', label='Average', linestyle='--')
    axs[0].set_title('Training Set Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_ylim(20, 100)  # 设置Y坐标轴范围
    axs[0].legend()

    # 验证集精度和平均精度
    for label, acc_values in val_accuracies.items():
        axs[1].plot(epochs[:len(acc_values)], acc_values, label=label)

    avg_val_values = [sum(values) / len(values) for values in zip(*val_accuracies.values())]
    axs[1].plot(epochs[:len(avg_val_values)], avg_val_values, marker='o', label='Average', linestyle='--')
    axs[1].set_title('Validation Set Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_ylim(20, 100)  # 设置Y坐标轴范围
    axs[1].legend()

    # 损失图
    min_length_all = min(len(epochs), len(train_loss), len(val_loss))
    axs[2].plot(epochs[:min_length_all], train_loss[:min_length_all], label='Training Loss')
    axs[2].plot(epochs[:min_length_all], val_loss[:min_length_all], label='Validation Loss')
    axs[2].set_title('Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()

    # 调整布局
    plt.tight_layout()

    # 保存图像
    log_name = log_path.split('\\')[-1].split('.')[0]
    save_path = os.path.join(current_folder, 'OutPuts', 'Plots', f"{log_name}_plots.png")
    plt.savefig(save_path)

# 获取当前脚本所在文件夹的路径
current_folder = os.path.dirname(os.path.abspath(__file__))
# 构建相对路径
logs_folder = os.path.join(current_folder, 'OutPuts', 'Log')

# 如果文件夹不存在，则创建
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

# 获取所有txt文件的列表
log_files = [f for f in os.listdir(logs_folder) if f.endswith('.txt')]

# 检查是否有.txt文件
if not log_files:
    print(f"No .txt files found in {logs_folder}. Exiting.")
    exit()

# 遍历所有日志文件
for log_file in log_files:
    log_path = os.path.join(logs_folder, log_file)

    # 调用函数生成图
    generate_plots(log_path)