# -*- coding: utf-8 -*-
# ==============================================================================
# Multi-Model Classifier - Training Logic
# 包含完整的模型训练、验证、测试及评估逻辑。
# Contains the complete logic for model training, validation, testing, and evaluation.
# ==============================================================================

import os
import sys
import time
import logging
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from Plotter import plot_comprehensive_results

def create_logger(model_name, output_dir):
    """
    创建日志器以保存训练日志。
    Create a logger to save training logs.
    """
    log_dir = os.path.join(output_dir, 'Log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{model_name}_log.txt')
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    fileHandler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')

    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader, 
                       num_epochs, learning_rate, device, class_names, output_dir):
    """
    核心训练与评估函数。
    Core training and evaluation function.
    """
    print(f"\n{'='*60}\n[Trainer] Starting Training (开始训练): {model_name}\n{'='*60}")
    model = model.to(device)
    logger = create_logger(model_name, output_dir)
    logger.info(f"Model: {model_name} | Epochs: {num_epochs} | LR: {learning_rate}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_acc = 0.0
    num_classes = len(class_names)
    
    # 记录每个类别的精度 (Track accuracy per class)
    train_acc_history = [[] for _ in range(num_classes)]
    val_acc_history = [[] for _ in range(num_classes)]
    test_acc_history = [[] for _ in range(num_classes)]
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # ==================== Training Phase ====================
        model.train()
        train_loss = 0.0
        correct_train = {i: 0 for i in range(num_classes)}
        total_train = {i: 0 for i in range(num_classes)}
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            for c in range(num_classes):
                class_idx = labels == c
                total_train[c] += class_idx.sum().item()
                correct_train[c] += (predicted[class_idx] == labels[class_idx]).sum().item()
                
        # 计算每个类别的训练精度并记录
        for c in range(num_classes):
            acc = 100. * correct_train[c] / total_train[c] if total_train[c] > 0 else 0
            train_acc_history[c].append(acc)
            
        avg_train_acc = np.mean([train_acc_history[c][-1] for c in range(num_classes)])
        logger.info(f"Train Loss (训练损失): {train_loss/len(train_loader):.4f} | Avg Train Acc (平均训练精度): {avg_train_acc:.2f}%")
        
        # ==================== Validation Phase ====================
        model.eval()
        val_loss = 0.0
        correct_val = {i: 0 for i in range(num_classes)}
        total_val = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                for c in range(num_classes):
                    class_idx = labels == c
                    total_val[c] += class_idx.sum().item()
                    correct_val[c] += (predicted[class_idx] == labels[class_idx]).sum().item()
                    
        # 计算每个类别的验证精度并记录
        for c in range(num_classes):
            acc = 100. * correct_val[c] / total_val[c] if total_val[c] > 0 else 0
            val_acc_history[c].append(acc)
            
        avg_val_acc = np.mean([val_acc_history[c][-1] for c in range(num_classes)])
        logger.info(f"Val Loss (验证损失): {val_loss/len(val_loader):.4f} | Avg Val Acc (平均验证精度): {avg_val_acc:.2f}%")
        
        scheduler.step()
        
        # Save best model (保存最佳模型)
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            weights_dir = os.path.join(output_dir, 'Weights')
            os.makedirs(weights_dir, exist_ok=True)
            best_model_path = os.path.join(weights_dir, f'{model_name}_Best.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"-> Best model saved (已保存最佳模型) with acc: {best_val_acc:.2f}%")

    # ==================== Testing Phase ====================
    logger.info("\n--- Testing Phase (测试阶段) ---")
    model.eval()
    all_labels, all_preds = [], []
    correct_test = {i: 0 for i in range(num_classes)}
    total_test = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            for c in range(num_classes):
                class_idx = labels == c
                total_test[c] += class_idx.sum().item()
                correct_test[c] += (predicted[class_idx] == labels[class_idx]).sum().item()
                
    for c in range(num_classes):
        acc = 100. * correct_test[c] / total_test[c] if total_test[c] > 0 else 0
        test_acc_history[c].append(acc)
        logger.info(f"Class '{class_names[c]}' Test Acc: {acc:.2f}%")
            
    avg_test_acc = np.mean([test_acc_history[c][-1] for c in range(num_classes)])
    logger.info(f"==> Average Test Accuracy (平均测试精度): {avg_test_acc:.2f}%")
    
    # Calculate Metrics (计算混淆矩阵和F1)
    conf_mat = confusion_matrix(all_labels, all_preds)
    f1_scores = f1_score(all_labels, all_preds, average=None)
    logger.info(f"Confusion Matrix (混淆矩阵):\n{conf_mat}")
    logger.info(f"F1 Scores: {f1_scores}")

    # Plotting (调用独立成图文件)
    plot_comprehensive_results(
        model_name, 
        train_acc_history, 
        val_acc_history, 
        test_acc_history, 
        conf_mat, 
        f1_scores, 
        avg_test_acc, 
        class_names,
        output_dir
    )
    
    total_time = (time.time() - start_time) / 60
    logger.info(f"\n[Trainer] Finished {model_name} in {total_time:.2f} mins.")
