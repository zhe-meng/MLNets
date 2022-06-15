'''
MLNet-A and MLNet-B.

Meng Z, Jiao L, Liang M, et al. Hyperspectral image classification 
with mixed link networks[J]. IEEE Journal of Selected Topics in Applied 
Earth Observations and Remote Sensing, 2021, 14: 2494-2507.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, k1=36, k2=36, dropRate=0):
        super(Bottleneck, self).__init__()
        # MLB-A
        if k1 > 0:
            planes = expansion * k1
            self.bn1_1 = nn.BatchNorm2d(inplanes)
            self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(planes)
            self.conv1_2 = nn.Conv2d(planes, k1, kernel_size=3, padding=1, bias=False)

        if k2 > 0:
            planes = expansion * k2
            self.bn2_1 = nn.BatchNorm2d(inplanes)
            self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(planes)
            self.conv2_2 = nn.Conv2d(planes, k2, kernel_size=3, padding=1, bias=False)

        self.dropRate = dropRate
        self.relu = nn.ReLU(inplace=True)
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        if self.k1 > 0:
            inner_link = self.bn1_1(x)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_1(inner_link)
            inner_link = self.bn1_2(inner_link)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_2(inner_link)

        if self.k2 > 0:
            outer_link = self.bn2_1(x)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_1(outer_link)
            outer_link = self.bn2_2(outer_link)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_2(outer_link)

        if self.dropRate > 0:
            inner_link = F.dropout(inner_link, p=self.dropRate, training=self.training)
            outer_link = F.dropout(outer_link, p=self.dropRate, training=self.training)

        c = x.size(1)
        if self.k1 > 0 and self.k1 < c:
            xl = x[:, 0: c - self.k1, :, :]
            xr = x[:, c - self.k1: c, :, :] + inner_link
            x = torch.cat((xl, xr), 1)
        elif self.k1 == c:
            x = x + inner_link

        if self.k2 > 0:
            out = torch.cat((x, outer_link), 1)
        else:
            out = x

        return out


class Bottleneck2(nn.Module):
    def __init__(self, inplanes, expansion=4, k1=36, k2=36, dropRate=0):
        super(Bottleneck2, self).__init__()
        # MLB-B
        if k1 > 0:
            planes = expansion * k1
            self.bn1_1 = nn.BatchNorm2d(inplanes)
            self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(planes)
            self.conv1_2 = nn.Conv2d(planes, k1, kernel_size=3, padding=1, bias=False)

        if k2 > 0:
            planes = expansion * k2
            self.bn2_1 = nn.BatchNorm2d(inplanes)
            self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(planes)
            self.conv2_2 = nn.Conv2d(planes, k2, kernel_size=3, padding=1, bias=False)

        self.dropRate = dropRate
        self.relu = nn.ReLU(inplace=True)
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        if self.k1 > 0:
            inner_link = self.bn1_1(x)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_1(inner_link)
            inner_link = self.bn1_2(inner_link)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_2(inner_link)

        if self.k2 > 0:
            outer_link = self.bn2_1(x)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_1(outer_link)
            outer_link = self.bn2_2(outer_link)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_2(outer_link)

        if self.dropRate > 0:
            inner_link = F.dropout(inner_link, p=self.dropRate, training=self.training)
            outer_link = F.dropout(outer_link, p=self.dropRate, training=self.training)

        if self.k2 > 0:
            out = torch.cat((x, outer_link), 1)
        else:
            out = x

        c = out.size(1)

        if self.k1 > 0 and self.k1 < c:
            xl = out[:, 0: c - self.k1, :, :]
            xr = out[:, c - self.k1: c, :, :] + inner_link
            out = torch.cat((xl, xr), 1)
        elif self.k1 == c:
            out = out + inner_link

        return out


class MLNet_A(nn.Module):
    def __init__(self, num_classes, channels, k1=36, k2=36, num_blcoks=3):
        super(MLNet_A, self).__init__()

        unit = Bottleneck  # MLB-A

        self.k1 = k1
        self.k2 = k2

        self.dropRate = 0

        self.inplanes = max(self.k1, self.k2 * 2)

        self.conv = nn.Conv2d(channels, self.inplanes, 3, 1, 1, bias=False)

        self.blocks = self._make_block(unit, num_blcoks)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, unit, unit_num):
        layers = []
        for _ in range(unit_num):

            layers.append(unit(self.inplanes, k1=self.k1, k2=self.k2, dropRate=self.dropRate))
            self.inplanes += self.k2

        return nn.Sequential(*layers)

    def forward(self, input):

        output = self.conv(input)
        output = self.blocks(output)
        output = self.relu(self.bn(output))

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output


class MLNet_B(nn.Module):
    def __init__(self, num_classes, channels, k1=36, k2=36, num_blcoks=3):
        super(MLNet_B, self).__init__()

        unit = Bottleneck2 # MLB-B

        self.k1 = k1
        self.k2 = k2

        self.dropRate = 0

        self.inplanes = max(self.k1, self.k2 * 2)

        self.conv = nn.Conv2d(channels, self.inplanes, 3, 1, 1, bias=False)

        self.blocks = self._make_block(unit, num_blcoks)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, unit, unit_num):
        layers = []
        for _ in range(unit_num):

            layers.append(unit(self.inplanes, k1=self.k1, k2=self.k2, dropRate=self.dropRate))
            self.inplanes += self.k2

        return nn.Sequential(*layers)

    def forward(self, input):

        output = self.conv(input)
        output = self.blocks(output)
        output = self.relu(self.bn(output))

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output


if __name__ == '__main__':

    model = MLNet_A(num_classes=16, channels=200)
    # model = MLNet_B(num_classes=16, channels=200)
    model.eval()
    print(model)
    input = torch.randn(100, 200, 11, 11)
    y = model(input)
    print(y.size())