# UNet Biomedical Image Segmentation

| Model:       | UNet           | UNet++           |
| -------------|:-------------:|:-------------:|
| Pytorch Code |[UNet Pytorch Code](https://github.com/Vinayak-VG/My-Projects/blob/main/Computer%20Vision%20Projects/U-Net%20Image%20Segmentation/U-Net/UNet.ipynb)| [UNet++ Pytorch Code](https://github.com/Vinayak-VG/My-Projects/blob/main/Computer%20Vision%20Projects/U-Net%20Image%20Segmentation/U-Net%2B%2B/UNet%2B%2B.ipynb)|
| Colab Link   |[UNet Colab Link](https://colab.research.google.com/drive/1G8ZBrbeFKVr7QOqfXsjmPY07OvRm1kaa?usp=sharing)      |   [UNet++ Colab Link](https://colab.research.google.com/drive/1TyBJHZRoVzZfwarTbzbeq3NyCHyPuqAy?usp=sharing)|
| Paper        | [UNet Paper](https://arxiv.org/pdf/1505.04597.pdf)      | [UNet++ Paper](https://arxiv.org/pdf/1807.10165.pdf) |

## UNet vs UNet++

| Model:        | UNet          | UNet++|
| ------------- |:-------------:| -----:|
| Loss          | 0.1176        | 0.0756|
| Accuracy      | 95.3%         | 97.8% |

As you can see UNet++ is clearly better when compared UNet. The dense convolution blocks and the skip pathways help to reduce the semantic gap between the encoder and the decoder part and hence the loss function is able to converge to the minima easily and hence better accuracy

## Setting Up

# install requirements
```
pip install elasticdeform
```

# Dataset
[Dataset](https://drive.google.com/drive/folders/1OWcrg0fSsm-vtoeJpeXOJ_VRTgf3JWsf?usp=sharing)

The data set is provided by the EM segmentation challenge that was started at ISBI 2012. The training data is a set of 30 images (512x512 pixels) from serial section transmission electron microscopy of the Drosophila first instar larva ventral nerve cord (VNC). Each image comes with a corresponding fully annotated ground truth segmentation map for cells (white) and membranes (black).









