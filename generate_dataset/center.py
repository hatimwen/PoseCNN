import numpy as np

# all positions of boxes from the vicon system need to be adjusted by v_offset
p1 = np.array([10, -10, 56])
p2 = np.array([10, 121, 10])
p3 = np.array([201, -10, -10])
offset = (p1 + p2 + p3) / 3
print(offset)


def get_box_center(vicon_input, box_size):
    box_corner = (vicon_input - offset)
    box_center = np.array([0, 0, 0])
    for i in range(3):
        box_center[i] = box_corner[i] + box_size[i]/2
    return box_center