import os


base_path = os.path.join(os.environ['thesis'], "config", "camera")
input_path = os.path.join(base_path, "results-cam-homesatcocatkin_wssrcthesisbagcalibration.txt")
with open(input_path, "r") as f:
    lines = f.readlines()
for line in lines:
    if "distortion:" in line:
        distortion = line.split("[")[1].split("]")[0].split()
        print(distortion)
for line in lines:
    if "projection:" in line:
        projection = line.split("[")[1].split("]")[0].split()
        print(projection)
output_path = os.path.join(base_path, "d415_hand_eye.yaml")

output_str = """distortion:
  parameters:
    rows: 4
    cols: 1
    data:
    - {}
    - {}
    - {}
    - {}
  type: radial-tangential
type: pinhole
label: d415_821312061282
line-delay-nanoseconds: 0
image_width: 640
image_height: 480
intrinsics:
  rows: 4
  cols: 1
  data:
  - {}
  - {}
  - {}
  - {}
"""
output_str = output_str.format(*(distortion + projection))

print(output_str)
with open(output_path, "w") as output:
    output.write(output_str)
