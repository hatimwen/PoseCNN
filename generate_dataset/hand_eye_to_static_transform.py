import os
import ast


base_path = os.path.join(os.environ['thesis'], "config", "camera")
input_path = os.path.join(base_path, "calibration_optimized.json")
with open(input_path, "r") as f:
    data = f.read()

data = ast.literal_eval(data)
translation = [data["translation"][key] for key in sorted(data["translation"])]
rotation = [data["rotation"][key] for key in sorted(data["rotation"])]

output_str = """<?xml version="1.0" encoding="utf-8"?>

<launch>
    <param name="use_sim_time"  value="true" />
    <node pkg="tf" type="static_transform_publisher" name="davis_to_camera" args="{} {} {} {} {} {} {}  davis camera 5">
    </node>
    <node pkg="tf" type="static_transform_publisher" name="tracker_to_corner" args="-0.07366666666666667, -0.033666666666666664, -0.018666666666666668 0 0 0 1 box_corner_tracker box_corner1 5">
    </node>
    <node pkg="tf" type="static_transform_publisher" name="corner_to_center" args="0.1745, 0.1065, 0.062 0 0 0 1 box_corner1 box1 5">
    </node>
    <node pkg="rosbag" type="play"  name="rosbag_play" args="--clock /home/satco/PycharmProjects/PoseCNN/bag/dataset_one_box.bag" output="screen">
    </node>
    <node pkg="rosbag" type="record" name="rosbag_record" args="record -O /home/satco/PycharmProjects/PoseCNN/bag/test.bag -a">
    </node>
</launch>"""
output_str = output_str.format(*(translation + rotation))
print(output_str)
output_path = os.path.join(os.environ['thesis'], "launch", "recording", "add_static_transform.launch")
with open(output_path, "w") as output:
    output.write(output_str)
