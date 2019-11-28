import torch
import torch.nn as nn
import torch.nn.functional as F

def res_unit(input_layer, nbF):
    batch_norm1 = nn.BatchNorm2d(input_layer.shape[0])
    part2 = F.relu(batch_norm1(input_layer))
    conv1 = nn.Conv2d(part2.shape[0], nbF, (3, 3))
    part3 = conv1(part2)
    batch_norm2 = nn.BatchNorm2d(part3.shape[0])
    part5 = F.relu(batch_norm2(part3))
    conv2 = nn.Conv2d(part5.shape[0], nbF, (3, 3))
    part6 = conv2(part5)
    output = input_layer + part6
    return output


class Encoder(torch.nn.Module):
    # shape for input x is (256, 256, 3)
    def __init__(self):
        super(Encoder, self).__init__()

        # Input channels = 3, output channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Input channels = 32, output channels = 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Input channels = 64, output channels = 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Input channels = 128, output channels = 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x):
        # Computes the activation of the first convolution
        x = self.conv1(x)
        x = res_unit(x, 32)
        x = F.relu(x)
        x = self.pool1(x)

        # Computes the activation of the second convolution
        x = self.conv2(x)
        x = res_unit(x, 64)
        x = F.relu(x)
        x = self.pool2(x)

        # Computes the activation of the first convolution
        x = self.conv1(x)
        x = res_unit(x, 128)
        x = F.relu(x)
        x = self.pool3(x)

        # Computes the activation of the first convolution
        x = self.conv1(x)
        x = res_unit(x, 256)
        x = F.relu(x)
        x = self.pool4(x)

        return x

