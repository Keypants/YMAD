import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from MAD import mad
from PIL import Image
import numpy as np
import os
import json
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ptflops import get_model_complexity_info

# Set random seed for reproducibility
torch.manual_seed(0)

dataset_path = 'F:/研究生/实验程序/数据集/Classifier_training_dataset'
unlabeled_male_path = os.path.join(dataset_path, 'male_crops1')
unlabeled_female_path = os.path.join(dataset_path, 'female_crops1')

# 图像大小
image_size = (256, 256)

# 数据预处理
def preprocess_image(image_path, image_size):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image) / 255.0
    image = (image - 0.5) / 0.5  # 归一化到 [-1, 1]
    return image

# Load data from class folders
def load_unlabeled_dataset():
    images = []
    labels = []
    file_paths = []

    # 加载未标注的male数据
    unlabeled_male_files = os.listdir(unlabeled_male_path)
    for file in unlabeled_male_files:
        if file.endswith('.jpg'):
            image_path = os.path.join(unlabeled_male_path, file)
            images.append(preprocess_image(image_path, image_size))
            labels.append(1)  # 伪标签
            file_paths.append(image_path)

    # 加载未标注的female数据
    unlabeled_female_files = os.listdir(unlabeled_female_path)
    for file in unlabeled_female_files:
        if file.endswith('.jpg'):
            image_path = os.path.join(unlabeled_female_path, file)
            images.append(preprocess_image(image_path, image_size))
            labels.append(0)  # 伪标签
            file_paths.append(image_path)

    return images, labels, file_paths

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, labels, file_paths):
        self.images = images
        self.labels = labels
        self.file_paths = file_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        return image, label, file_path

# 加载有标注的数据集
images, labels, file_paths = load_unlabeled_dataset()

# 转换为张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = torch.stack([torch.tensor(image).permute(2, 0, 1).to(device).float() for image in images])
labels = torch.tensor(labels).to(device)

# 创建数据集对象
dataset = CustomDataset(images, labels, file_paths)

# 划分训练集、验证集和测试集
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 8
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Build MobileNetV2 model
model = mad(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to the available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train model
num_epochs = 30
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0
best_model_wts = None
best_epoch = 0


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for images, labels, _ in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_data_loader.dataset)
    train_accuracy = train_correct.double() / len(train_data_loader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy.item())

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels, _ in val_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_data_loader.dataset)
    val_accuracy = val_correct.double() / len(val_data_loader.dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy.item())

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = model.state_dict().copy()
        best_epoch = epoch


    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Function to evaluate the model
# Evaluate the best model
def evaluate_model(model_wts):
    model.load_state_dict(model_wts)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    all_preds = []
    all_labels = []
    all_file_paths = []

    with torch.no_grad():
        for images, labels, file_paths in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_file_paths.extend(file_paths)

    test_loss = test_loss / len(test_data_loader.dataset)
    test_accuracy = test_correct.double() / len(test_data_loader.dataset)
    test_f1_score = f1_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_conf_matrix = confusion_matrix(all_labels, all_preds)

    return test_loss, test_accuracy, test_f1_score, test_recall, test_conf_matrix, all_file_paths, all_preds

# Save the best model
best_model_path = './trained_classifier/best_model_MAD.pth'
torch.save(best_model_wts, best_model_path)

# # Save the last model
# last_model_path = './训练好的分类器/last_model_mobile.pth'
# torch.save(last_model_wts, last_model_path)

# Load and evaluate the best model
best_model_wts = torch.load(best_model_path)
print("Evaluating the best model...")
best_test_loss, best_test_accuracy, best_test_f1_score, best_test_recall, best_test_conf_matrix, best_all_file_paths, best_all_preds = evaluate_model(best_model_wts)
print(f'Best Epoch: {best_epoch + 1}')
print(f'Best Model Test Loss: {best_test_loss:.4f}')
print(f'Best Model Test Accuracy: {best_test_accuracy:.4f}')
print(f'Best Model Test F1 Score: {best_test_f1_score:.4f}')
print(f'Best Model Test Recall: {best_test_recall:.4f}')
print('Best Model Confusion Matrix:')
macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print(best_test_conf_matrix)


# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=3)
plt.plot(val_losses, label='Val Loss', linewidth=3)
plt.legend(fontsize=18, prop={'family': 'Times New Roman'})
plt.xlabel('Epoch', fontsize=26, fontname="Times New Roman")
plt.ylabel('Loss', fontsize=26, fontname="Times New Roman")
plt.title('Training and Validation Loss', fontsize=26, fontname="Times New Roman")
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xticks(fontsize=22, fontname="Times New Roman")
plt.yticks(fontsize=22, fontname="Times New Roman")

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', linewidth=3)
plt.plot(val_accuracies, label='Val Accuracy', linewidth=3)
plt.legend(fontsize=18, prop={'family': 'Times New Roman'})
plt.xlabel('Epoch', fontsize=26, fontname="Times New Roman")
plt.ylabel('Accuracy', fontsize=26, fontname="Times New Roman")
plt.title('Training and Validation Accuracy', fontsize=26, fontname="Times New Roman")
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xticks(fontsize=22, fontname="Times New Roman")
plt.yticks(fontsize=22, fontname="Times New Roman")

plt.tight_layout()
plt.show()

# Output file names and prediction results for the best model
print("Best Model Predictions:")
for file_path, pred in zip(best_all_file_paths, best_all_preds):
    print(f'File: {file_path}, Predicted Class: {pred}')


# Plot normalized confusion matrix for the best model (percentage form)
plt.figure(figsize=(10, 7))
best_test_conf_matrix_normalized = best_test_conf_matrix.astype('float') / best_test_conf_matrix.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(best_test_conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 30, "fontname":"Times New Roman"})
plt.xlabel('Predicted', fontsize=26, fontname="Times New Roman")
plt.ylabel('True', fontsize=26, fontname="Times New Roman")
plt.title("Ours Normalized Confusion Matrix", fontsize=28, fontname="Times New Roman")

plt.xticks(fontsize=22, fontname="Times New Roman")
plt.yticks(fontsize=22, fontname="Times New Roman")

cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=22)
for label in cbar.ax.get_yticklabels():
    label.set_fontname("Times New Roman")

plt.show()


