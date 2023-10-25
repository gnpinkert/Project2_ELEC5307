'''
this script is for the network of Project 2.

You can change any parts of this code

-------------------------------------------
'''
import torch.nn as nn
import torch
import torch.hub
import torchvision.transforms
from torchvision.models import resnet50


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.patch_size = 14
        self.fc1 = nn.Linear(384, 196)
        self.fc2 = nn.Linear(196, 25)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch, channel, height, width = x.size()

        next_largest_height_divisor = ((height // self.patch_size) + 1) * self.patch_size
        next_largest_width_divisor = ((width // self.patch_size) + 1) * self.patch_size

        resize_op = torchvision.transforms.Resize(size=(next_largest_height_divisor, next_largest_width_divisor),antialias=True)
        x = resize_op(x)
        x = self.backbone(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class ResNetwork(nn.Module):
    def __init__(self):
        super(ResNetwork, self).__init__()
        self.backbone = resnet50(weights="IMAGENET1K_V1")
        self.fc1 = nn.Linear(1000, 196)
        self.fc2 = nn.Linear(196, 25)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


if __name__=="__main__":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    testImage = torch.rand((1,3,256,256))
    testmodel = Network()

    testOutput = testmodel(testImage)
    print(testOutput.shape)


