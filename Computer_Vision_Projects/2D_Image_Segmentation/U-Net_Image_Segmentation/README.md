# UNet Biomedical Image Segmentation

| Model:       | UNet           | UNet++           |
| -------------|:-------------:|:-------------:|
| Pytorch Code |[UNet Pytorch Code](https://github.com/Vinayak-VG/My-Projects/blob/main/Computer%20Vision%20Projects/2D%20Image%20Segmentation/U-Net%20Image%20Segmentation/U-Net/UNet.ipynb)| [UNet++ Pytorch Code](https://github.com/Vinayak-VG/My-Projects/blob/main/Computer%20Vision%20Projects/2D%20Image%20Segmentation/U-Net%20Image%20Segmentation/U-Net%2B%2B/UNet%2B%2B.ipynb)|
| Colab Link   |[UNet Colab Link](https://colab.research.google.com/drive/1G8ZBrbeFKVr7QOqfXsjmPY07OvRm1kaa?usp=sharing)      |   [UNet++ Colab Link](https://colab.research.google.com/drive/1TyBJHZRoVzZfwarTbzbeq3NyCHyPuqAy?usp=sharing)|
| Paper        | [UNet Paper](https://arxiv.org/pdf/1505.04597.pdf)      | [UNet++ Paper](https://arxiv.org/pdf/1807.10165.pdf) |
| Report       | [UNet Report](https://github.com/Vinayak-VG/My-Projects/files/6740030/U-Net_.Convolutional.Networks.for.Biomedical.Image.Segmentation.pdf)   | [UNet++ Report](https://github.com/Vinayak-VG/My-Projects/files/6740032/UNet%2B%2B_.A.Nested.U-Net.Architecture.for.Medical.Image.Segmentation.pdf)   | 



## UNet vs UNet++

| Model:        | UNet          | UNet++|
| ------------- |:-------------:| -----:|
| Loss          | 0.1176        | 0.0756|
| Accuracy      | 95.3%         | 97.8% |

As you can see UNet++ is clearly better when compared UNet. The dense convolution blocks and the skip pathways help to reduce the semantic gap between the encoder and the decoder part and hence the loss function is able to converge to the minima easily and hence better accuracy


![caption](https://user-images.githubusercontent.com/80670240/124690672-2c372800-def8-11eb-9880-b2803f34de2a.gif)


The image on the left represents serial section transmission electron microscopy of the Drosophila first instar larva ventral nerve cord (VNC). The image on the right is the segmentation map of the image on the left predicted by the U-Net Model

# Setting Up

## Install Requirements
```
pip install elasticdeform
```

## Dataset
[Dataset](https://drive.google.com/drive/folders/1OWcrg0fSsm-vtoeJpeXOJ_VRTgf3JWsf?usp=sharing)

The data set is provided by the EM segmentation challenge that was started at ISBI 2012. The training data is a set of 30 images (512x512 pixels) from serial section transmission electron microscopy of the Drosophila first instar larva ventral nerve cord (VNC). Each image comes with a corresponding fully annotated ground truth segmentation map for cells (white) and membranes (black).

---

[Vinayak Gupta](https://github.com/Vinayak-VG)
1st June 2021






