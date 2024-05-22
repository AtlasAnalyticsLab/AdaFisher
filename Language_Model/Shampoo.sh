#!/bin/bash
python run_exp.py --layers 4 --batch_size 32 --lr 0.003 --weight_decay 0.1 --log 1 --epochs 12 --optimizer Shampoo \
       --damping 1e-12 --momentum 0.9 --curvature_update_interval 1 --ema_decay -1 --device cuda