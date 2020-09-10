import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch import nn
import matplotlib.pyplot as plt


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


# LeNet-5
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            View((-1, 64 * 4 * 4)),
            nn.Linear(64 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(500, 10),
            # nn.Softmax(dim=1),
        )

        # 重みの初期化(weight init)
        for m in self.layers.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.layers(x)


# GPU or CPU の判別
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# modelの定義
model = cnn().to(device)
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

