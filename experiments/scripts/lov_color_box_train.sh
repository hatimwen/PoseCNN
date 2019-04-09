#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_box_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/mnt/drive_c/datasets/kaju/tcmalloc_libs/libtcmalloc.so.4
#--ckpt /mnt/drive_c/datasets/kaju/PoseCNN/output/lov/lov_000_box_train/run577/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_797_epoch_5.ckpt \
#--ckpt /mnt/drive_c/datasets/kaju/PoseCNN/output/lov/lov_000_box_train/run454/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_8_epoch_108.ckpt \
# --weights data/imagenet_models/vgg16.npy \
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16.npy \
  --ckpt /mnt/drive_c/datasets/kaju/PoseCNN/output/lov/lov_000_box_train/run612/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_797_epoch_5.ckpt \
  --rand \
  --imdb lov_single_000_box_train \
  --cfg experiments/cfgs/lov_color_box.yml \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --iters 160000
