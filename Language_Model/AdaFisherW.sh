#!/bin/bash
python run_exp.py --layers 4 --batch_size 32 --weight_decay 0.1 --lr 0.0001 --log 1 --epochs 50 --optimizer AdaFisherW \
       --damping 1e-3 --gamma1 0.92 --gamma2 0.008 --device cuda