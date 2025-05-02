import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from res_model1 import resnet18  # 从res_model1模块导入resnet18模型
from sklearn.model_selection import GroupShuffleSplit
import re

# 定义主函数，包含模型的训练流程
def main(csv_file_path, image_dir_path, num_epochs=100, batch_size=4, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取CSV文件并计算标签的均值和标准差
    data = pd.read_csv(csv_file_path)
    label_means = data[['aperture_radius', 'inner_radius', 'outer_radius']].mean()
    label_stds = data[['aperture_radius', 'inner_radius', 'outer_radius']].std()

    # 保存标签的均值和标准差
    label_means.to_csv('label_means.csv', header=False, index=True)
    label_stds.to_csv('label_stds.csv', header=False, index=True)

    # 自定义数据集类
    class RadiusDataset(Dataset):
        def __init__(self, csv_file, image_dir, transform=None, label_means=None, label_stds=None):
            self.data = pd.read_csv(csv_file)
            self.image_dir = image_dir
            self.transform = transform

            # 从文件名提取 star_id（兼容两种格式）
            self.data['star_id'] = self.data['cutout_filename'].apply(
                lambda x: int(re.search(r'star_(\d+)', x).group(1))
            )

            self.label_means = torch.tensor(label_means.values, dtype=torch.float32)
            self.label_stds = torch.tensor(label_stds.values, dtype=torch.float32)


        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.image_dir, self.data.iloc[idx]['cutout_filename'])
            file_extension = os.path.splitext(img_name)[1]

            if file_extension == '.fits':
                with fits.open(img_name) as hdul:
                    image_data = hdul[0].data.astype('float32')

                    min_positive = np.min(image_data[image_data > 0]) if np.any(image_data > 0) else 1e-6
                    image_data[image_data <= 0] = min_positive

                    image_data = np.log(image_data)
                    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
                    image_data = (image_data * 255).astype('uint8')

                    image = Image.fromarray(image_data)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")

            aperture_radius = self.data.iloc[idx]['aperture_radius']
            inner_radius = self.data.iloc[idx]['inner_radius']
            outer_radius = self.data.iloc[idx]['outer_radius']

            log_aperture_radius = np.log(aperture_radius + 1e-5)
            log_inner_radius = np.log(inner_radius + 1e-5)
            log_outer_radius = np.log(outer_radius + 1e-5)

            labels = torch.tensor([log_aperture_radius, log_inner_radius, log_outer_radius], dtype=torch.float32)
            labels = (labels - self.label_means) / self.label_stds

            star_id = self.data.iloc[idx]['star_id']  # 获取 star_id

            if self.transform:
                image = self.transform(image)  # 应用transform

            return image, labels, star_id  # 返回的是Tensor类型的图像

        # 修改后的自定义数据集划分逻辑，确保旋转图像和原图出现在相同集合中

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

    # 验证划分结果是否无重叠
    def validate_split(train_dataset, test_dataset):
        train_star_ids = set(train_dataset.dataset.data.iloc[train_dataset.indices]['star_id'])
        test_star_ids = set(test_dataset.dataset.data.iloc[test_dataset.indices]['star_id'])
        overlap = train_star_ids.intersection(test_star_ids)
        if overlap:
            raise ValueError(f"错误: {len(overlap)} 个 star_id 同时出现在训练集和测试集")
        else:
            print("验证通过：训练集与测试集无重叠 star_id")

    # 数据预处理变换
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转图像
        transforms.RandomRotation(15),  # 随机旋转图像
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
    ])

    # 创建数据集对象并划分数据集
    dataset = RadiusDataset(csv_file=csv_file_path, image_dir=image_dir_path, transform=transform,label_means=label_means,
                            label_stds=label_stds)
    train_dataset, test_dataset = custom_train_test_split(dataset, test_size=0.3)

    # —— 在这里打印数据集大小 ——
    total_size = len(dataset)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print(f"总数据量: {total_size}, 训练集: {train_size}, 测试集: {test_size}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = resnet18()  # 使用自定义的ResNet18模型
    model.to(device)  # 将模型加载到设备上
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 使用Adam优化器，设置学习率和权重衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率调度器，每10个epoch学习率减半

    best_test_loss = float('inf')
    train_losses, test_losses = [], []

    # 开始训练模型
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for images, labels, star_ids in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据加载到设备上

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播，获得模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累计损失

        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)  # 记录训练损失

        # 测试模型
        model.eval()  # 设置模型为评估模式
        test_loss = 0.0
        all_predictions = []
        all_true_labels = []
        all_star_ids = []  # 用于保存测试集的 star_id
        with torch.no_grad():  # 不计算梯度，节省内存
            for images, labels, star_ids in test_loader:
                images, labels = images.to(device), labels.to(device)  # 加载测试数据
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                test_loss += loss.item()  # 累计测试损失

                all_predictions.append(outputs.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())

                # 将当前批次的 star_id 加入列表
                all_star_ids.extend(star_ids.cpu().numpy())  # 注意这里的 star_ids 是从 DataLoader 里返回的

        # 保存 star_id 到 CSV 文件
        star_ids_df = pd.DataFrame(all_star_ids, columns=['star_id'])
        # 如果real_star_id列存在，可以提取真实的star_id，否则直接使用star_id
        if 'real_star_id' in dataset.data.columns:
            star_ids_df['real_star_id'] = star_ids_df['star_id'].apply(
                lambda x: dataset.data.loc[dataset.data['star_id'] == x, 'real_star_id'].values[0]
            )
        else:
            # 如果real_star_id列不存在，直接将star_id作为真实star_id
            star_ids_df['real_star_id'] = star_ids_df['star_id']
        star_ids_df.to_csv(r'D:\juanji\11.3\planb\data\11.12\test_star_ids.csv', index=False)

        test_loss /= len(test_loader)  # 计算平均测试损失
        test_losses.append(test_loss)  # 记录测试损失

        # 计算MSE
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_true_labels = np.concatenate(all_true_labels, axis=0)

        mse = np.mean((all_predictions - all_true_labels) ** 2)

        # 保存MSE结果到CSV文件
        mse_df = pd.DataFrame({
            'Predictions': all_predictions.flatten(),
            'True Labels': all_true_labels.flatten(),
            'Squared Error': (all_predictions.flatten() - all_true_labels.flatten()) ** 2
        })

        mse_df.to_csv(r'D:\juanji\11.3\planb\data\11.12\mse_results.csv', index=False)

        # 保存最优模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss  # 更新最优测试损失
            torch.save(model.state_dict(),
                       r'D:\juanji\11.3\planb\data\11.12\resnet_best_model18.pth')  # 保存模型参数

        scheduler.step()  # 更新学习率

        # 打印当前epoch的训练和测试损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.5f}, Test Loss: {test_loss:.5f}")

    # 保存训练过程数据到CSV文件
    training_data = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Test Loss': test_losses
    })
    if not os.path.exists('output'):
        os.makedirs('output')  # 如果输出目录不存在，创建目录
    training_data.to_csv('training_process.csv', index=False)  # 保存训练过程数据

    print("Model training completed and saved.")

    # 创建一个图形用于绘制损失曲线，设置图像大小为 ViT 的 2 倍
    fig1, ax1 = plt.subplots(figsize=(24, 16))  # 将图像大小设为 ViT 的两倍

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
    plt.savefig(r'D:\juanji\11.21\11.12\loss_plot_resnet.png')
    plt.show()

# 主程序入口
if __name__ == "__main__":
    csv_file_path = r"D:\juanji\11.3\planb\data\11.12\expanded_dataset预测数据集.csv"  # CSV文件路径
    image_dir_path = r"D:\juanji\11.3\planb\data\11.12\cutout"  # 图像目录路径
    main(csv_file_path, image_dir_path)  # 调用主函数开始训练