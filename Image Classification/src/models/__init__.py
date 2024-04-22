from models.resnet import (resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,
                               wide_resnet50_2, wide_resnet101_2)
from models.resnet_cifar import (ResNet18 as resnet18_cifar, ResNet34 as resnet34_cifar, ResNet50 as resnet50_cifar,
                                     ResNet101 as resnet101_cifar, ResNet152 as resnet152_cifar)
from models.densenet import (densenet121, densenet161, densenet169, densenet201)
from models.densenet_cifar import (DenseNet121 as densenet121_cifar, DenseNet161 as densenet161_cifar,
                                       DenseNet169 as densenet169_cifar, DenseNet201 as densenet201_cifar)
from models.efficientnet_cifar import efficientnet_B0 as efficientnet_B0_cifar
from models.vit import s_ViT
from models.simple_convnet import simple_convnet as simple_convnet_cifar
from models.swin import (swin_t, swin_s, swin_b, swin_l)
from models.deepvit import deepvit
from models.cct import cct
from models.focalnet import FocalNet
from models.mobilenetv3 import mobilenetv3

def get_network(name: str, num_classes: int, **kwargs) -> None:
    return \
        simple_convnet_cifar(num_classes) if name == "simple_convnetCifar" else\
        resnet18(
            num_classes=num_classes) if name == 'resnet18' else\
        resnet34(
            num_classes=num_classes) if name == 'resnet34' else\
        resnet50(
            num_classes=num_classes) if name == 'resnet50' else\
        resnet101(
            num_classes=num_classes) if name == 'resnet101' else\
        resnet152(
            num_classes=num_classes) if name == 'resnet152' else\
        resnext50_32x4d(
            num_classes=num_classes) if name == 'resnext50' else\
        resnext101_32x8d(
            num_classes=num_classes) if name == 'resnext101' else\
        wide_resnet50_2(
            num_classes=num_classes) if name == 'wide_resnet50' else\
        wide_resnet101_2(
            num_classes=num_classes) if name == 'wide_resnet101' else\
        resnet18_cifar(
            num_classes=num_classes) if name == 'resnet18Cifar' else\
        resnet34_cifar(
            num_classes=num_classes) if name == 'resnet34Cifar' else\
        resnet50_cifar(
            num_classes=num_classes) if name == 'resnet50Cifar' else\
        resnet101_cifar(
            num_classes=num_classes) if name == 'resnet101Cifar' else \
        resnet152_cifar(
            num_classes=num_classes) if name == 'resnet152Cifar' else \
        densenet121(
            num_classes=num_classes) if name == 'densenet121' else\
        densenet161(
            num_classes=num_classes) if name == 'densenet161' else\
        densenet169(
            num_classes=num_classes) if name == 'densenet169' else\
        densenet201(
            num_classes=num_classes) if name == 'densenet201' else \
        densenet121_cifar(
            num_classes=num_classes) if name == 'densenet121Cifar' else \
        densenet161_cifar(
            num_classes=num_classes) if name == 'densenet161Cifar' else \
        densenet169_cifar(
            num_classes=num_classes) if name == 'densenet169Cifar' else \
        densenet201_cifar(
            num_classes=num_classes) if name == 'densenet201Cifar' else\
        efficientnet_B0_cifar(
            num_classes=num_classes) if name == 'efficientnet_B0Cifar' else\
        s_ViT(
            num_classes=num_classes, **kwargs) if name == 's_ViT' else \
        swin_t(
            num_classes=num_classes, **kwargs) if name == 'swin_t' else\
        swin_s(
            num_classes=num_classes, **kwargs) if name == 'swin_s' else\
        swin_b(
            num_classes=num_classes, **kwargs) if name == 'swin_b' else\
        swin_l(
            num_classes=num_classes, **kwargs) if name == 'swin_l' else \
        deepvit(
            num_classes=num_classes, **kwargs) if name == 'deepvit' else \
        cct(
            num_classes=num_classes, **kwargs) if name == 'cct' else \
        FocalNet(
            num_classes=num_classes, **kwargs) if name == 'focalnet' else \
        mobilenetv3(
            num_classes=num_classes, **kwargs) if name == 'mobilenetv3' else \
        None