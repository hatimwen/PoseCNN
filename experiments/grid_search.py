import subprocess
from scipy import arange


def main():
    config = '''EXP_DIR: lov
INPUT: COLOR
TRAIN:
  SINGLE_FRAME: True
  TRAINABLE: True
  WEIGHT_REG: 0.0001
  LEARNING_RATE: {}
  MOMENTUM: 0.9
  GAMMA: 0.1
  STEPSIZE: 1000
  SYMSIZE: 0
  SCALES_BASE: !!python/tuple [1.0]
  IMS_PER_BATCH: 2
  NUM_CLASSES: 2
  NUM_UNITS: 64
  SNAPSHOT_ITERS: 1000
  SNAPSHOT_INFIX: lov_box
  SNAPSHOT_PREFIX: vgg16_fcn_color_single_frame_2d_pose_add_sym
  USE_FLIPPED: False
  CHROMATIC: True
  ADD_NOISE: True
  VOTING_THRESHOLD: 100
  VERTEX_REG_2D: True
  VERTEX_REG_3D: False
  VERTEX_W: {}
  POSE_W: {}
  VISUALIZE: False
  POSE_REG: True
  SYNTHESIZE: False
  SYN_RATIO: 10
  SYN_ONLINE: True
  SYN_CLASS_INDEX: 15
  SYNROOT: 'data/LOV/data_syn_036_box/'
  SYNNUM: 10000
  THRESHOLD_LABEL: 1.0
TEST:
  SINGLE_FRAME: True
  SCALES_BASE: !!python/tuple [1.0]
  VERTEX_REG_2D: True
  VERTEX_REG_3D: False
  VISUALIZE: True
  POSE_REG: True
  POSE_REFINE: False
  SYNTHETIC: False
'''

    # min, max, stepsize
    lr_intverals = [0.00001, 0.0001 + 0.00001, 0.00003]
    pose_wheight_intverals = [1, 3, 1]
    vertex_wheight_intverals = [3, 7, 2]
    counter = 0
    for lr in arange(*lr_intverals):
        for pose_wheight in arange(*pose_wheight_intverals):
            for vertex_wheight in arange(*vertex_wheight_intverals):
                config_out = open("../experiments/cfgs/lov_color_box.yml", "w")
                config_formatted = config.format(lr, float(vertex_wheight), float(pose_wheight))
                print("Run {}:\nlr: {}\nvertex_wheight: {}\npose_wheight: {}".format(counter, lr, vertex_wheight, pose_wheight))
                config_out.write(config_formatted)
                config_out.close()
                counter += 1
                subprocess.call(['./../experiments/scripts/lov_color_box_train.sh 0'], env={"PATH": "/home/satco/remote_home/envs/python3.6/bin/"})


if __name__ == '__main__':
    main()
