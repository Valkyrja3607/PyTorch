import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
import matplotlib.pyplot as plt



#データ読み込み
bs=128 #batch size

#正規化                                                             
transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

trainset=MNIST(root='./data',train=True,download=True,transform=transform)
trainloader=DataLoader(trainset,batch_size=bs,shuffle=True)

testset=MNIST(root='./data',train=False,download=True,transform=transform)
testloader=DataLoader(testset,batch_size=bs,shuffle=False)

#LeNet-5
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,10),
            #nn.Softmax(dim=1),
        )

        #重みの初期化(weight init)
        for m in self.layers.children():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        return self.layers(x)


# GPU or CPU の判別                                                                 
device="cuda" if torch.cuda.is_available() else "cpu"

# modelの定義                                                                          
model=cnn().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
criterion=nn.CrossEntropyLoss(reduction="mean")

#train
print("train")
model=model.train()
loss_train_list=[]
for i in range(3):
    for j,(x,y) in enumerate(trainloader):
        x=x.to(device)
        #体格行列でone-hotに変換
        #y=torch.eye(10)[y].to(device)

        #推定
        predict=model.forward(x)

        #loss(bachの平均)
        loss=criterion(predict,y)
        #eps=1e-7
        #loss=-torch.mean(y*torch.log(predict+eps))
        loss_train_list.append(loss.item())
        #勾配初期化
        optimizer.zero_grad()
        #勾配計算(backward)
        loss.backward()
        #パラメータ更新
        optimizer.step()

        #進捗報告
        if j%100==0:
            print("%03depoch, %05d, loss=%.5f"%(i,j,loss.item()))

#test
print("test")
model=model.eval()
total,tp=0,0
for (x,y) in testloader:
    x=x.to(device)
    #推定
    predict=model.forward(x)
    pre=predict.argmax(1).to('cpu') #one-hotからスカラ―値

    #answer count
    total+=y.shape[0]
    tp+=(y==pre).sum().item()

#acc
acc=tp/total
print("test accuracy=%.3f"%acc)

plt.plot(loss_train_list)
plt.show()


