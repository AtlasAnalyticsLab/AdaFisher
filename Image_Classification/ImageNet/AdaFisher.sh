#!/bin/bash
# ADD YOUR SETTINGS HERE
python main.py --arch resnet50 --workers 8 --epochs 90 --lr 0.001 --optimizer AdaFisher --data pathtoImageNet