
##########################DDAG


CUDA_VISIBLE_DEVICES=1 python CMPS_attack.py --data /sda1/data -s sysu_v2  --batch_size 64  --resume /sda1/DDAG/save_model/sysu_G_P_3_drop_0.2_4_8_lr_0.1_seed_0_best.t  --max-eps 8 