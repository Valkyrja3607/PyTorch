import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SE(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, se_ratio):
        super(Block, self).__init__()
        # 1x1
        w_b = int(round(w_out * bottleneck_ratio))
        self.conv1 = nn.Conv2d(w_in, w_b, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w_b)
        # 3x3
        num_groups = w_b // group_width
        self.conv2 = nn.Conv2d(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=num_groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(w_b)
        # se
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = SE(w_b, w_se)
        # 1x1
        self.conv3 = nn.Conv2d(w_b, w_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(w_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.with_se:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RegNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(RegNet, self).__init__()
        self.cfg = cfg
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(0)
        self.layer2 = self._make_layer(1)
        self.layer3 = self._make_layer(2)
        self.layer4 = self._make_layer(3)
        self.linear = nn.Linear(self.cfg["widths"][-1], num_classes)

    def _make_layer(self, idx):
        depth = self.cfg["depths"][idx]
        width = self.cfg["widths"][idx]
        stride = self.cfg["strides"][idx]
        group_width = self.cfg["group_width"]
        bottleneck_ratio = self.cfg["bottleneck_ratio"]
        se_ratio = self.cfg["se_ratio"]

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(
                Block(self.in_planes, width, s, group_width, bottleneck_ratio, se_ratio)
            )
            self.in_planes = width
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RegNetX_200MF():
    cfg = {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "strides": [1, 1, 2, 2],
        "group_width": 8,
        "bottleneck_ratio": 1,
        "se_ratio": 0,
    }
    return RegNet(cfg)


batch_size = 128
data_path = "./data"

# 正規化
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5,], std=[0.5, 0.5, 0.5]),
    ]
)

trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

num_images = 50000
num_val = 5000
budget = 2500
initial_budget = 5000
num_classes = 10


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


# resnet
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.resnet = models.resnet50()
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


model = RegNetX_200MF()

# GPU or CPU の判別
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# modelの定義
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(reduction="mean")

# train
print("train")
model = model.train()
loss_train_list = []
acc_train_list = []
total, tp = 0, 0
for i in range(20):
    acc_count = 0
    for j, (x, y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)
        # 体格行列でone-hotに変換
        # y=torch.eye(10)[y].to(device)

        # 推定
        predict = model.forward(x)

        # loss(bachの平均)
        loss = criterion(predict, y)
        # eps=1e-7
        # loss=-torch.mean(y*torch.log(predict+eps))
        loss_train_list.append(loss.item())
        # acc
        pre = predict.argmax(1).to(device)
        total += y.shape[0]
        tp += (y == pre).sum().item()
        # 勾配初期化
        optimizer.zero_grad()
        # 勾配計算(backward)
        loss.backward()
        # パラメータ更新
        optimizer.step()

        # 進捗報告
        if j % 100 == 0:
            print(
                "%03depoch, %05d, loss=%.5f, acc=%.5f" % (i, j, loss.item(), tp / total)
            )
    acc = tp / total
    acc_train_list.append(acc)

plt.plot(acc_train_list)
plt.show()
plt.plot(loss_train_list)
plt.show()
# test
print("test")
model = model.eval()
total, tp = 0, 0
for (x, y) in testloader:
    x = x.to(device)
    # 推定
    predict = model.forward(x)
    pre = predict.argmax(1).to("cpu")  # one-hotからスカラ―値

    # answer count
    total += y.shape[0]
    tp += (y == pre).sum().item()

# acc
acc = tp / total
print("test accuracy=%.3f" % acc)

plt.plot(loss_train_list)
plt.show()

