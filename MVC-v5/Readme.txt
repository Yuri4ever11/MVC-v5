MCV-v5 (Multi-Model Vision Classifier-CNN Version5)
2024/8/24

版本更新(大改)：
	通用性提升： 该版本经过全面升级，具备更强的通用性，可应用于更广泛的场景。
	注释完善： 代码中添加了大量注释，方便用户理解代码逻辑和功能。
	代码优化： 代码结构更加简洁，易于阅读和维护。
	新手友好： 保留了强大的功能性的同时，对新手用户更加友好，降低了使用门槛。
++添加Image_Classification_with_Weights，可以根据训练好的模型对数据进行单独识别，设置置信度，提高数据纯度。
优先看新版本，注释更多 更容易理解。

##第一次写这些东西，不是很规范，很多名称的定义很中二，代码也有一些冗余的地方，如果吐槽还请口下留情。##


1. 日志记录：
create_logger(model_name) 函数用于创建日志记录器，它将日志信息输出到控制台和指定的文件中。
global_logger 用于记录全局信息，每个模型训练时会创建一个新的日志记录器，用于记录该模型的训练过程。

2. 模型定义：
models 列表包含了要训练的多个模型，每个模型都是一个类的实例，例如 C1_LeNet5(num_classes=4)。
model_names 列表包含了每个模型的名称，用于标识不同的模型。

3. 数据预处理：
transform 用于对图像进行预处理，包括灰度化、缩放、转换为张量和归一化。
代码中提供了两种预处理方式，分别适用于单通道和三通道的图像。

4. 数据加载：
train_dataset, val_dataset, test_dataset 分别加载训练集、验证集和测试集。
train_loader, val_loader, test_loader 分别创建数据加载器，用于批次读取数据。

5. 模型训练：
device 用于指定模型运行的设备，可以选择 CPU 或 GPU。
num_epochs 设置训练的轮数。
criterion 设置损失函数，代码中提供了多种损失函数，例如交叉熵损失、Focal Loss、OhemCELoss 等。
optimizer 设置优化器，代码中使用了 Adam 优化器。
scheduler 设置学习率衰减策略，代码中使用了余弦退火策略。
循环遍历 models 和 model_names 列表，对每个模型进行训练。

6. 模型评估：
train_acc_history, val_acc_history, test_acc_history 用于记录每个类别的精度。
average_train, average_val, average_test 用于记录每个模型的平均精度。
train_loss_history, val_loss_history, test_loss_history 用于记录每个模型的损失值。

代码解释：
代码首先定义了日志记录器，用于记录训练过程中的信息。
然后定义了要训练的模型列表和名称列表，并设置了数据预处理和数据加载方式。
接着，代码循环遍历每个模型，对每个模型进行训练和评估。
训练过程中，代码使用了不同的损失函数、优化器和学习率衰减策略，并记录了每个模型的精度和损失值。

代码特点：
代码使用 PyTorch 框架实现，并使用了一些常用的深度学习库，例如 torchvision 和 torch.optim。
代码结构清晰，注释详细，易于理解和维护。
代码支持多模型训练，并提供了多种损失函数和学习率衰减策略，方便用户根据实际情况进行选择。