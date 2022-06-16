# Hyperspectral Image Classification With Mixed Link Networks

PyTorch implementation of hyperspectral image classification with mixed link networks.

![1](fig/1.png)
![1](fig/2.png)
![1](fig/3.png)

# Basic Usage

```
import torch
from MLNets import MLNet_A, MLNet_B

model = MLNet_A(num_classes=16, channels=200)
# model = MLNet_B(num_classes=16, channels=200)
model.eval()
print(model)
input = torch.randn(100, 200, 11, 11)
y = model(input)
print(y.size())
```

# Paper

[Hyperspectral image classification with mixed link networks](https://ieeexplore.ieee.org/abstract/document/9335013)

Please cite our paper if you find it useful for your research.

```
@article{meng2021hyperspectral,
  title={Hyperspectral image classification with mixed link networks},
  author={Meng, Zhe and Jiao, Licheng and Liang, Miaomiao and Zhao, Feng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={14},
  pages={2494--2507},
  year={2021},
  publisher={IEEE}
}
```
# Reference Paper

* Mixed Link Networks https://arxiv.org/abs/1802.01808

# Reference Code

* MixNet https://github.com/DeepInsight-PCALab/MixNet

