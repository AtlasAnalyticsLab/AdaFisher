#!/bin/bash
python run_exp.py --layers 4 --batch_size 32 --lr 0.00001 --weight_decay 0.1 --log 1 --epochs 55 --optimizer adamw \
       --device cuda