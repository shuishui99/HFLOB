#TODO: CNN
import torch
from torchvision import transforms
from torchvision import  datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.30381,))])
train_dataset = datasets.MNIST(root='../data/mnist/',train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
test_dataset = datasets.MNIST(root='../data/mnist/', train=False, download=True, transform=transform)
test_load = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# design model

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
        # flaten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        return self.fc(x)


model = Model()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2,momentum=0.5)


def train(epoch):
    """
    :param epoch:
    """
    running_loss = 0.0
    for batch_id, (inputs, targets) in enumerate(train_loader, 0):
        optimizer.zero_grad() # 对网络中当前参数的导数置0
        ouputs = model(inputs) #网络向前计算 forward
        loss = criterion(ouputs, targets) #计算loss
        loss.backward() #backforward
        optimizer.step() #更新参数

        running_loss += loss.item()
        if batch_id % 300 == 299:
            print('[%d, %5d] loss :%.3f' %(epoch+1, batch_id+1, running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_load:
            images, lables = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += lables.size(0)
            correct += (predicted == lables).sum().item()
    print('accuracy on test set: %d %%' % (100*(correct/total)))



if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()



