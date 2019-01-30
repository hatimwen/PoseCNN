from common import invert_quat

infra1_to_infra1_optical = [-0.5, 0.5, -0.5, 0.5]
infra1_to_camera = [0, 0, 0, 1]
camera_to_camera_color = [-0.0110191544518, -0.00340897939168, 0.00536672864109, 0.999919056892]
camera_color_to_camera_optical = [-0.5, 0.5, -0.5, 0.5]
print(invert_quat(infra1_to_infra1_optical))
print(invert_quat(infra1_to_camera))
print(invert_quat(camera_to_camera_color))
print(invert_quat(camera_color_to_camera_optical))
