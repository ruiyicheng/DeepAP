import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from VIT1 import VisionTransformer  # 确保 VisionTransformer 在 VIT1.py 中定义
import time  # 导入 time 库
from tqdm import tqdm  # 确保导入的是 tqdm 函数
import re
from sklearn.model_selection import GroupShuffleSplit

# 自定义数据集
class RadiusDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # 从文件名提取 star_id（兼容 cutout_star_27.fits 或 star_27_subimage60.fits）
        self.data['star_id'] = self.data['cutout_filename'].apply(
            lambda x: int(re.search(r'star_(\d+)', x).group(1))
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['cutout_filename'])
        file_extension = os.path.splitext(img_name)[1]

        try:
            if file_extension == '.fits':
                with fits.open(img_name) as hdul:
                    image_data = hdul[0].data.astype('float32')
                    min_positive = np.min(image_data[image_data > 0]) if np.any(image_data > 0) else 1e-6
                    image_data[image_data <= 0] = min_positive
                    image_data = np.log(image_data)
                    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
                    image_data = (image_data * 255).astype('uint8')
                    image = Image.fromarray(image_data)
                image = image.convert('L')  # Convert to grayscale
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: {e} - Skipping file: {img_name}")
            # 返回一个空的图像和标签，跳过该样本
            empty_image = Image.new('L', (256, 256))  # 使用空白图像代替
            return empty_image, torch.zeros(1, dtype=torch.float32)

        # 直接获取分类标签 (polluted)
        polluted_label = float(self.data.iloc[idx]['polluted'])  # 0 or 1

        # 创建分类标签张量（现在只包含 'polluted' 标签）
        class_labels = torch.tensor([polluted_label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        star_id = self.data.iloc[idx]['star_id']  # 获取 star_id
        # 返回图像和分类标签
        return image, class_labels, star_id  # 返回三个值

def custom_train_test_split(dataset, test_size=0.3):
    # 提取所有样本的 star_id 列表
    star_ids = dataset.data['star_id'].tolist()

    # 使用 GroupShuffleSplit 按 star_id 分组划分
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(dataset.data, groups=star_ids))

    # 创建训练集和测试集
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, test_dataset

def main(csv_file_path, image_dir_path, num_epochs=50, batch_size=256, lr_classification=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转图像
        transforms.RandomRotation(15),  # 随机旋转图像
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
    ])

    # 创建数据集
    dataset = RadiusDataset(csv_file=csv_file_path, image_dir=image_dir_path, transform=transform)
    train_dataset, test_dataset = custom_train_test_split(dataset, test_size=0.3)
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建 DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 初始化模型、损失函数和优化器
    model = VisionTransformer(num_classes=1, num_heads=8, num_blocks=12)  # 分类任务类别数量设为 2
    model.to(device)

    # 定义分类损失函数
    criterion_classification = nn.BCEWithLogitsLoss()

    # 创建优化器
    optimizer_classification = optim.Adam(model.parameters(), lr=lr_classification, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer_classification, step_size=10, gamma=0.5)

    best_test_loss = float('inf')
    train_losses, test_losses = [], []
    test_accuracies = []

    # 开始训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_classification_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, data in loop:
            images, classification_labels, star_ids = data
            images = images.to(device)
            classification_labels = classification_labels.to(device)

            # 清零梯度
            optimizer_classification.zero_grad()

            # 前向传播
            out = model(images)

            # 计算分类损失
            loss_classification = criterion_classification(out, classification_labels)

            # 反向传播并更新参数
            loss_classification.backward()
            optimizer_classification.step()

            running_classification_loss += loss_classification.item()
            running_loss += loss_classification.item()
            loop.set_description(f'epoch:[{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss_classification.item())

        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        epoch_classification_loss = running_classification_loss / len(train_loader)

        train_losses.append(epoch_loss)

        # 打印训练损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.5f}")

        # 测试模型
        model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 记录测试开始时间
        start_time = time.time()

        all_star_ids = []  # 用于保存测试集的 star_id

        with torch.no_grad():
            for images, classification_labels, star_ids in test_loader:
                images = images.to(device)
                classification_labels = classification_labels.to(device)

                out = model(images)

                loss_classification = criterion_classification(out, classification_labels)

                test_loss += loss_classification.item()

                # 计算预测结果并评估准确率
                probabilities = torch.sigmoid(out)  # 获取 sigmoid 概率值
                predicted_labels = (probabilities >= 0.5).float()  # 预测为 0 或 1

                # 比较预测和真实标签
                correct_predictions += (predicted_labels == classification_labels).sum().item()
                total_samples += classification_labels.numel()  # 计算总样本数

                # 收集所有的 star_id
                all_star_ids.extend(star_ids.cpu().numpy())  # 收集当前批次的 star_id

        # 记录测试结束时间
        end_time = time.time()
        test_duration = end_time - start_time  # 计算测试时间

        # 计算平均测试损失
        test_loss /= len(test_loader)

        # 计算准确率
        accuracy = (correct_predictions / total_samples) * 100

        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        # 打印测试阶段损失和准确率
        print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {accuracy:.2f}%, Test Time: {test_duration:.2f}s")

        # 保存最优模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if not os.path.exists('output'):
                os.makedirs('output')
            torch.save(model.state_dict(), 'D:/juanji/11.21/output12.5/VIT_model02.pth')

        scheduler.step()  # 更新学习率

    # 训练完成后，在测试集上评估模型
    print("开始测试阶段...")
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_true_labels = []

    # 记录测试开始时间
    start_time = time.time()

    all_star_ids = []  # 用于保存测试集的 star_id

    star_times = []  # 用于保存每个星体的处理时间

    with torch.no_grad():
        for images, classification_labels, star_ids in test_loader:
            images = images.to(device)
            classification_labels = classification_labels.to(device)

            # 开始记录每个星体的时间
            for i in range(len(star_ids)):
                star_id = star_ids[i]
                star_start_time = time.time()  # 开始时间

                out = model(images[i].unsqueeze(0))  # 处理单个图像
                # 记录处理时间
                star_end_time = time.time()
                star_process_time = star_end_time - star_start_time

                # 将 star_id 和其处理时间保存
                all_star_ids.append(star_id)
                star_times.append(star_process_time)

            out = model(images)

            loss_classification = criterion_classification(out, classification_labels)

            test_loss += loss_classification.item()

            probabilities = torch.sigmoid(out)
            all_predictions.append(probabilities.cpu().numpy())
            all_true_labels.append(classification_labels.cpu().numpy())

            predicted_labels = (probabilities >= 0.5).float()

            correct_predictions += (predicted_labels == classification_labels).sum().item()
            total_samples += classification_labels.numel()

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    test_loss /= len(test_loader)
    test_accuracy = (correct_predictions / total_samples) * 100

    # 记录测试结束时间
    end_time = time.time()
    test_duration = end_time - start_time  # 计算测试时间

    print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_accuracy:.2f}%, Test Time: {test_duration:.2f}s")

    # 保存测试集的 star_id 到 CSV 文件
    star_ids_df = pd.DataFrame(all_star_ids, columns=['star_id'])
    star_ids_df.to_csv(r'D:\juanji\11.21\11.12\test_star_ids.csv', index=False)

    # 保存预测结果
    predictions_df = pd.DataFrame(all_predictions, columns=['Polluted_Probability'])
    true_labels_df = pd.DataFrame(all_true_labels, columns=['Polluted_Label']).astype(int)
    results_df = pd.concat([predictions_df, true_labels_df], axis=1)

    results_df.to_csv(r'D:\juanji\11.21\概率结果\test_predictions.csv', index=False)

    print("Model training completed and saved.")

    # 打印测试阶段损失和准确率
    print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_accuracy:.2f}%")

    # 保存训练过程数据
    training_data = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Test Accuracy': test_accuracies
    })
    if not os.path.exists('output'):
        os.makedirs('output')
    training_data.to_csv(r'D:\juanji\11.21\11.12\training_process_classification2.csv', index=False)

    # 保存每个星体的处理时间到 CSV 文件
    star_times_df = pd.DataFrame({
        'star_id': all_star_ids,
        'processing_time': star_times
    })
    star_times_df.to_csv(r'D:\juanji\11.21\11.12\star_processing_times.csv', index=False)

    print("Model training completed and saved.")

    # ============================ 绘制训练和验证损失曲线 ==========================
    # 创建一个图形用于绘制损失曲线
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制训练损失和测试损失
    ax1.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='orange')

    # 设置纵轴为对数坐标
    ax1.set_yscale('log')  # 将损失的纵轴设为对数坐标

    # 设置坐标轴标签和标题
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train and Test Loss')

    # 显示图例
    ax1.legend()

    # 保存损失图
    plt.savefig(r'D:\juanji\11.21\11.12\loss_plot.png')
    plt.show()

    # ============================ 绘制准确率图 ==========================
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # 绘制测试准确率
    ax2.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='purple', linestyle='--')

    # 设置坐标轴标签和标题
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')

    # 显示图例
    ax2.legend()

    # 保存准确率图
    plt.savefig(r'D:\juanji\11.21\11.12\accuracy_plot.png')
    plt.show()


if __name__ == "__main__":
    csv_file_path = r"D:\juanji\11.21\11.12\expanded_dataset分类数据集.csv"  # CSV 文件路径
    image_dir_path = r"D:\juanji\11.21\11.12\cutout"  # 输入图片的路径
    main(csv_file_path, image_dir_path)    # 运行主函数
