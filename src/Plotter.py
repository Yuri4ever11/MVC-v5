# -*- coding: utf-8 -*-
# ==============================================================================
# Multi-Model Classifier - Visualization Tools
# 此文件包含所有成图相关的工具代码，用于绘制精度曲线和混淆矩阵。
# Contains visualization tools to plot accuracy curves and confusion matrices.
# ==============================================================================

import os
import matplotlib.pyplot as plt

def plot_comprehensive_results(model_name, train_acc_history, val_acc_history, test_acc_history, 
                               conf_matrix, f1_scores, avg_test_acc, class_names, output_dir='OutPuts'):
    """
    Plots a comprehensive view including accuracy per class and overall confusion matrix info.
    保留了原版多图排版，优化了美观度，支持中英双语显示。
    """
    num_classes = len(class_names)
    rows = (num_classes // 2) + 1
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    
    # 设置全局字体和样式 (Set global font and style)
    # 支持中文显示的字体回退列表：黑体, 微软雅黑, 苹方, 以及系统的非衬线字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
    plt.rcParams['font.size'] = 12
    
    # 将 axes 展平，方便索引 (Flatten axes for easy indexing)
    if rows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()
    
    # 绘制每个类别的训练、验证和测试精度曲线
    for i in range(num_classes):
        ax = axes_flat[i]
        
        epochs = range(1, len(train_acc_history[i]) + 1)
        ax.plot(epochs, train_acc_history[i], label='Train Acc', linewidth=2, linestyle='-', marker='o', alpha=0.8)
        ax.plot(epochs, val_acc_history[i], label='Val Acc', linewidth=2, linestyle='-', marker='s', alpha=0.8)
        
        ax.set_xlabel('Epochs (轮数)')
        ax.set_ylabel('Accuracy (准确率 %)')
        ax.set_title(f'Class: {class_names[i]}\nTrain: {train_acc_history[i][-1]:.2f}% | Val: {val_acc_history[i][-1]:.2f}%')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # 在最后一个使用过的子图后面紧接着展示混淆矩阵、F1分数和平均测试精度
    ax_text = axes_flat[num_classes]
    
    # 格式化 F1 分数显示
    f1_text = " | ".join([f"{class_names[j]}: {f1_scores[j]:.2f}" for j in range(num_classes)])
    
    confusion_text = (
        f"=== Model: {model_name} ===\n\n"
        f"Confusion Matrix (混淆矩阵):\n{conf_matrix}\n\n"
        f"F1 Scores:\n{f1_text}\n\n"
        f"Average Test Accuracy (平均测试精度): {avg_test_acc:.2f}%"
    )
    
    ax_text.text(0.5, 0.5, confusion_text, wrap=True, horizontalalignment='center', verticalalignment='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    ax_text.axis('off')
    
    # 如果有更多的空余子图，将它们隐藏
    for i in range(num_classes + 1, len(axes_flat)):
        axes_flat[i].axis('off')
        
    fig.tight_layout()
    
    # 保存图片 (Save image)
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{model_name}_Results.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Plotter] Comprehensive plot saved at (综合结果图已保存至): {plot_path}")
