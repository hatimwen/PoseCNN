#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

LOG="experiments/logs/lov_color_box_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./ros/test_ros_bag.py --gpu 0 \
  --bag bag/dataset5.6.bag \
  --network vgg16_convs \
  --model trained_nets/run150/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_497_epoch_4.ckpt \
  --imdb lov_single_000_box_train \
  --cfg experiments/cfgs/lov_color_box.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl \
