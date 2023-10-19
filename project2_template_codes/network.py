'''
this script is for the network of Project 2.

You can change any parts of this code

-------------------------------------------
'''
import torch.nn as nn
import torch
import torch.hub
import torchvision.transforms


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.patch_size = 14
        self.fc1 = nn.Linear(384, 196)
        self.fc2 = nn.Linear(196, 25)
        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):
        batch, channel, height, width = x.size()

        next_largest_height_divisor = ((height // self.patch_size) + 1) * self.patch_size
        next_largest_width_divisor = ((width // self.patch_size) + 1) * self.patch_size

        resize_op = torchvision.transforms.Resize(size=(next_largest_height_divisor, next_largest_width_divisor),antialias=True)
        x = resize_op(x)
        x = self.dinov2_vits14(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__=="__main__":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    testImage = torch.rand((1,3,256,256))
    testmodel = Network()

    testOutput = testmodel(testImage)
    print(testOutput.shape)


