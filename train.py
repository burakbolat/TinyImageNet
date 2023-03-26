import torch
import torch.nn as nn
import data
from model import ResNet

from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    trainset = data.TrainTinyImageNet("tiny-imagenet-200/train")
    train_loader = DataLoader(trainset, batch_size=256)

    model = ResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # Weight decay can be hired.

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(running_loss/(i+1))
        print("Epoch loss", running_loss)