import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

cfgs = {"vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
ranges = {"vgg16": ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}

'''Build VGG16 Network Configuration'''


def makeLayers(cfgs, batchnormal=False):
    layers = []
    in_chs = 3
    for val in cfgs:
        if val == 'M':
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        else:
            layers.append(nn.Conv2d(in_channels=in_chs, out_channels=val, kernel_size=(3, 3), padding=(1, 1)))
            if batchnormal:
                layers.append(nn.BatchNorm2d(num_features=val))
            layers.append(nn.ReLU(inplace=True))
            in_chs = val
    return nn.Sequential(*layers)


print(makeLayers(cfgs["vgg16"]))


class VGGNet(nn.Module):
    def __init__(self, layers):
        super(VGGNet, self).__init__()
        self.layers = layers  # VGG16 Configuration
        self.ranges = ranges["vgg16"]
        # self._init_weights_()

    def _init_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = {}
        for i in range(len(self.ranges)):
            for layer in range(self.ranges[i][0], self.ranges[i][1]):
                x = self.layers[layer](x)
            out['x%d' % (i + 1)] = x  # output of each maximum pooling layer
        # x5: torch.Size([5, 512, 5, 5])
        # x4: torch.Size([5, 512, 10, 10])
        # x3: torch.Size([5, 256, 20, 20])
        # x2: torch.Size([5, 128, 40, 40])
        # x1: torch.Size([5, 64, 80, 80])
        return out


class FCN32s(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes
        self.base_model = base_model  # VGG16
        self.conv = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=(1, 1))
        self.act = nn.ReLU(inplace=True)
        self.deconv32 = nn.ConvTranspose2d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(64, 64),
            stride=(32, 32),
            padding=(16, 16))  # (5 - 1) * 32 + 64 - 2 * 16 = 160

    def forward(self, x):
        out = self.base_model(x)
        x5 = out['x5']  # [5, 512, 5, 5]
        score = self.act(self.conv(x5))  # [5, 512, 5, 5] -> [5, 20, 5, 5]
        score = self.deconv32(score)  # [5, 20, 5, 5] -> [5, 20, 160, 160]
        return score


class FCN16s(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FCN16s, self).__init__()
        self.num_classes = num_classes
        self.base_model = base_model
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1)
        )  # (input - 1) * 2 + 4 - 2 * 1 =  2 * input
        self.deconv16 = nn.ConvTranspose2d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(32, 32),
            stride=(16, 16),
            padding=(8, 8)
        )  # (input - 1) * 16 + 32 - 2 * 8 = 16 * input
        self.bn = nn.BatchNorm2d(num_features=self.num_classes)

    def forward(self, x):
        out = self.base_model(x)
        x5 = out['x5']  # [5, 512, 5, 5]
        x4 = out['x4']  # [5, 512, 10, 10]
        score = self.act(self.conv(x5))  # [5, 512, 5, 5] -> [5, 20, 5, 5]
        score = self.bn(self.deconv2(score) + self.act(self.conv(x4)))
        # x4: [5, 512, 10, 10] -> [5, 20, 10, 10], score: [5, 20, 5, 5] -> [5, 20, 10, 10]
        score = self.deconv16(score)  # [5, 20, 10, 10] -> [5, 20, 160, 160]
        return score


class FCN8s(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        self.base_model = base_model
        self.act = nn.ReLU(inplace=True)
        '''conv: chs:in_chs -> chs:num_classes'''
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=(1, 1))
        self.deconv2_1 = nn.ConvTranspose2d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1)
        )  # (input - 1) * 2 + 4 - 2 * 1 =  2 * input
        self.deconv2_2 = nn.ConvTranspose2d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1)
        )  # (input - 1) * 2 + 4 - 2 * 1 =  2 * input
        self.deconv8 = nn.ConvTranspose2d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(16, 16),
            stride=(8, 8),
            padding=(4, 4)
        )  # (input - 1) * 8 + 16 - 2 * 4 =  16 * input
        self.bn1 = nn.BatchNorm2d(num_features=self.num_classes)
        self.bn2 = nn.BatchNorm2d(num_features=self.num_classes)

    def forward(self, x):
        out = self.base_model(x)
        x5 = out['x5']  # [5, 512, 5, 5]
        x4 = out['x4']  # [5, 512, 10, 10]
        x3 = out['x3']  # [5, 256, 20, 20]
        score = self.act(self.conv1(x5))  # [5, 512, 5, 5] -> [5, 20, 5, 5]
        score = self.bn1(self.deconv2_1(score) + self.act(self.conv2(x4)))
        # score:[5, 20, 5, 5] -> [5, 20, 10, 10], x4:[5, 512, 10, 10] -> [5, 20, 10, 10]
        score = self.bn2(self.deconv2_2(score) + self.act(self.conv3(x3)))
        # score:[5, 20, 10, 10] -> [5, 20, 20, 20], x3:[5, 256, 20, 20] -> [5, 20, 20, 20]
        score = self.deconv8(score)  # [5, 20, 20, 20] -> [5, 20, 160, 160]
        return score


model = VGGNet(makeLayers(cfgs["vgg16"]))
x = torch.randn((5, 3, 160, 160))
y = torch.randn((5, 20, 160, 160))
fcn = FCN32s(model, 20)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(fcn.parameters(), lr=1e-3, momentum=0.9)

for i in range(100):
    out = fcn(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("iter:{},   loss:{}".format(i, loss.data.item()))
