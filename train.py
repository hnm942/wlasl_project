import numpy as np
import pandas as pd
# for develop
import sys
import torch 
import torch.nn as nn
from tqdm import tqdm
import sys
sys.path.append("/workspace/src/efficientnet")
from datasets.czech_slr_dataset import CzechSLRDataset
from torchvision import transforms
from datasets.gaussian_noise import GaussianNoise
import torchvision
# from efficientnet_pytorch import EfficientNetmax_landmark_size
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import torchvision.models as models


# class CustomEfficientNet(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomEfficientNet, self).__init__()
#         self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=2)
#         num_ftrs = self.efficientnet._fc.in_features
#         self.efficientnet._fc = nn.Linear(num_ftrs, num_classes)
        
#     def forward(self, x):
#         return self.efficientnet(x)

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes, in_channels=2):
#         super(SimpleCNN, self).__init__()
        
#         # Khởi tạo các layer convolutional
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.fc = nn.Linear(1920, num_classes)
        
        
#         # self.resnet = models.resnet18(pretrained=True)
#         # self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # # Fully connected layer
#         # self.resnet.fc = nn.Linear(512, num_classes)
        
#     def forward(self, x):
#         # Forward qua các lớp convolutional
#         # print(x.shape)
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = nn.functional.max_pool2d(x, kernel_size=2)
#         # print(x.shape)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = nn.functional.max_pool2d(x, kernel_size=2)
#         # print(x.shape)
#         x = self.conv3(x)
#         x = nn.functional.relu(x)
#         x = nn.functional.max_pool2d(x, kernel_size=2)
#         x = self.conv4(x)
#         x = nn.functional.relu(x)
#         x = nn.functional.max_pool2d(x, kernel_size=2)
#         # print(x.shape)
#         # Flatten tensor
#         x = x.reshape(x.size(0), -1)
#         # print(x.shape)
#         # Forward qua fully connected layer
#         x = self.fc(x)
#         return x
#         return self.resnet(x)

class SimpleCNN1D(nn.Module):
    def __init__(self, num_classes, in_channels=2):
        super(SimpleCNN1D, self).__init__()

        # Khởi tạo các layer convolutional
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(17280, num_classes)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(8320, num_classes)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(3840, num_classes)



        # self.resnet = models.resnet18(pretrained=True)
        # self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Fully connected layer
        # self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward qua các lớp convolutional
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.conv1(x)
        # print(x.shape)

        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2)
        x = self.dropout1(x)
        x = x.view(x.size(0), x.size(1), -1, 54)
        # x = x.reshape(x.size(0), -1)
        # print(x.shape)
        # x = self.fc1(x)
        # print(x.shape)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.dropout2(x)
        # print(x.shape)w
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.dropout3(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc2(x)
        # x = self.conv3(x)
        # x = nn.functional.relu(x)
        # x = nn.functional.max_pool2d(x, kernel_size=2)
        # print(x.shape)
        # Flatten tensor
        # x = x.reshape(x.size(0), -1)
        # print(x.shape)
        # Forward qua fully connected layer
        # x = self.fc2(x)
        return x

# Định nghĩa residual block cho ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection nếu số kênh đầu vào khác số kênh đầu ra
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += self.shortcut(residual)
        x = self.relu(x)
        return x

# Định nghĩa mô hình ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Các lớp residual block
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Khởi tạo mô hình ResNet


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        

        inputs, labels = data
        labels = labels.view(-1)
        # print(inputs.shape)
        inputs = inputs.to(device).to(torch.float)
        inputs = inputs.permute(0, 3, 1, 2)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=1))) == int(labels[0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        scheduler.step(running_loss.item() / len(dataloader))
    return running_loss, pred_correct, pred_all, (pred_correct / pred_all)
def evaluate(model, dataloader, device, print_stats=False):

    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device).to(torch.float)
        inputs = inputs.permute(0, 3, 1, 2)
        labels = labels.view(-1)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs)
        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=1))) == int(labels[0]):
            stats[int(labels[0])][0] += 1
            pred_correct += 1

        stats[int(labels[0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return pred_correct, pred_all, (pred_correct / pred_all)


epochs = 300
num_classes = 101
gaussian_mean = 0
gaussian_std = 0.001 
training_set_path = "/workspace/data/wlasl_100/WLASL100_train_25fps.csv"
val_set_path = "/workspace/data/wlasl_100/WLASL100_val_25fps.csv"
lr = 0.001
save_checkpoints = True
seed = 100
scheduler_factor = 0.1
scheduler_patience = 0.01
log_freq = 1


g = torch.Generator()
g.manual_seed(seed)
device = torch.device("cuda:0") if torch.cuda.is_available() else  torch.device("cpu")
experiment_name = "output"
transform = transforms.Compose([GaussianNoise(gaussian_mean, gaussian_std)])
train_dataset = CzechSLRDataset(training_set_path, transform=transform, augmentations=True)
train_loader = DataLoader(train_dataset, shuffle=True, generator=g)
# train_dataset = CzechSLRDataset(training_set_path)
val_dataset = CzechSLRDataset(val_set_path)
val_loader = DataLoader(val_dataset, shuffle=True, generator=g)
model = SimpleCNN1D(num_classes)
model.to(device)
experiment_name
cel_criterion = nn.CrossEntropyLoss()
sgd_optimizer = optim.SGD(model.parameters(), lr= lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=scheduler_factor, patience=scheduler_patience)



train_acc, val_acc = 0, 0
losses, train_accs, val_accs = [], [], []
lr_progress = []
top_train_acc, top_val_acc = 0, 0
checkpoint_index = 0
# for param in slrt_model.transformer.parameters():
#     param.requires_grad = False

# for i, data in enumerate(train_loader):
#     inputs, labels = data
#     print(inputs.shape)


for epoch in range(epochs):
    train_loss, _, _, train_acc = train_epoch(model, train_loader, cel_criterion, sgd_optimizer, device)
    losses.append(train_loss.item() / len(train_loader))
    train_accs.append(train_acc)
    if val_loader:
        model.train(False)
        _, _, val_acc = evaluate(model, val_loader, device)
        model.train(True)
        val_accs.append(val_acc)

    # Save checkpoints if they are best in the current subset
    if save_checkpoints:
        if train_acc > top_train_acc:
            top_train_acc = train_acc
            torch.save(model, "out-checkpoints/" + experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

        if val_acc > top_val_acc:
            top_val_acc = val_acc
            torch.save(model, "out-checkpoints/" + experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

    if epoch % log_freq == 0:
        print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))
        logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))

        if val_loader:
            print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
            logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

        print("")
        logging.info("")

    # Reset the top accuracies on static subsets
    if epoch % 10 == 0:
        top_train_acc, top_val_acc = 0, 0
        checkpoint_index += 1

    lr_progress.append(sgd_optimizer.param_groups[0]["lr"])
