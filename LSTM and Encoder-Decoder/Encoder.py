import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair

# Parameters
lr = 0.00003
training_iters = 5
batch_size = 16
nbFilter=32
n_classes = 2 # manipulated vs unmanipulated
outSize=16
bSize = batch_size
device =  "cuda" if torch.cuda.is_available() else "cpu"

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

def res_unit(input_layer, nbF):
    batch_norm1 = nn.BatchNorm2d(input_layer.shape[1])
    batch_norm1 = batch_norm1.to(device)
    part1 = batch_norm1(input_layer)
    part2 = F.relu(part1)
    conv1 = Conv2d(part2.shape[1], nbF, (3, 3))
    conv1 = conv1.to(device)
    part3 = conv1(part2)
    batch_norm2 = nn.BatchNorm2d(part3.shape[1])
    batch_norm2 = batch_norm2.to(device)
    part4 = batch_norm2(part3)
    part5 = F.relu(part4)
    conv2 = Conv2d(part5.shape[1], nbF, (3, 3))
    conv2 = conv2.to(device)
    part6 = conv2(part5)
    output = input_layer + part6
    return output


class Encoder(torch.nn.Module):
    # shape for input x is (256, 256, 3)
    def __init__(self):
        super(Encoder, self).__init__()

        # Input channels = 3, output channels = 32
        self.conv1 = Conv2d(3, 32, kernel_size=(3, 3), stride=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Input channels = 32, output channels = 64
        self.conv2 = Conv2d(32, 64, kernel_size=(3, 3), stride=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Input channels = 64, output channels = 128
        self.conv3 = Conv2d(64, 128, kernel_size=(3, 3), stride=1, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Input channels = 128, output channels = 256
        self.conv4 = Conv2d(128, 256, kernel_size=(3, 3), stride=1, bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)


    def forward(self, x):
        # Computes the activation of the first convolution
        x = self.conv1(x)
        # batch_norm = nn.BatchNorm2d(x.shape[1])
        # x = batch_norm(x)
        x = res_unit(x, 32)
        x = F.relu(x)
        x = self.pool1(x)

        # Computes the activation of the second convolution
        x = self.conv2(x)
        # batch_norm = nn.BatchNorm2d(x.shape[1])
        # x = batch_norm(x)
        x = res_unit(x, 64)
        x = F.relu(x)
        x = self.pool2(x)

        # Computes the activation of the first convolution
        x = self.conv3(x)
        # batch_norm = nn.BatchNorm2d(x.shape[1])
        # x = batch_norm(x)
        x = res_unit(x, 128)
        x = F.relu(x)
        x = self.pool3(x)

        # Computes the activation of the first convolution
        x = self.conv4(x)
        # batch_norm = nn.BatchNorm2d(x.shape[1])
        # x = batch_norm(x)
        x = res_unit(x, 256)
        x = F.relu(x)
        x = self.pool4(x)

        return x