import os
import yaml


def main():
    folder_configs = "/home/satco/catkin_ws/src/thesis/config/datasets"
    out_folder = "/home/satco/catkin_ws/src/thesis/launch/recording/static_transforms"
    configs = os.listdir(folder_configs)
    for config in configs:
        dataset = config[:-5]
        print(dataset)
        with open(os.path.join(folder_configs, config), "r") as fin, open(os.path.join(out_folder, dataset + ".launch"),
                                                                          "w") as fout:
            header = '''<?xml version="1.0" encoding="utf-8"?>

<launch>
    '''
            fout.write(header)
            config = yaml.load(fin)
            boxes = config["boxes"]
            for i, box in enumerate(boxes):
                try:
                    box_str = '''<arg name="x{}"  default="$(eval {}/2)" />
    <arg name="y{}"  default="$(eval {}/2)" />
    <arg name="z{}"  default="$(eval -{}/2)" />
    
    '''.format(i+1, box[0]['x'], i+1, box[1]['y'], i+1, box[2]['z'])
                except IndexError:
                    print(boxes)
                fout.write(box_str)

            camera_transform = '''<!-- Davis to camera -->
    <node pkg="tf" type="static_transform_publisher" name="davis_to_camera" args="-0.048882992201385346 -0.00072006751937278676 0.076358135916286152 -0.20723877362797402 0.11935847973033537 -0.56413581317953265 0.79028882587944005 davis camera 5" />
    <!-- Corner tracker to corner -->
    <node pkg="tf" type="static_transform_publisher" name="tracker_to_corner1" args="-0.07366666666666667 -0.033666666666666664 -0.018666666666666668 0 0 0 1 box_corner_tracker box_corner 5" />
    <!-- Boxes center -->
    '''
            fout.write(camera_transform)
            for i, box in enumerate(boxes):
                if i == len(boxes) - 1:
                    box_str = '''<node pkg="tf" type="static_transform_publisher" name="corner{}_to_center{}" args="$(arg x{}) $(arg y{}) $(arg z{}) 0 0 0 1 box_corner box{} 5" />
'''.format(i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1)
                else:
                    box_str = '''<node pkg="tf" type="static_transform_publisher" name="corner{}_to_center{}" args="$(arg x{}) $(arg y{}) $(arg z{}) 0 0 0 1 box_corner box{} 5" />
    '''.format(i+1, i+1, i+1, i+1, i+1, i+1, i+1)
                fout.write(box_str)

            fout.write("</launch>\n")


if __name__ == "__main__":
    main()
