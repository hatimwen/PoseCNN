import ast
import bpy
from common import ros_to_blender_quat, create_dataset_folder, get_filename_prefix
import os
import time
import yaml


class Timer(object):  # pragma: no cover
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


# size (x,y,z) location (x,y,z)
def add_cube(size, location, quat, i, color):
    bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=location,
                                    layers=(True, False, False, False, False, False, False, False, False, False, False,
                                            False, False, False, False, False, False, False, False, False))
    bpy.context.object.name = "Cube" + str(i)
    bpy.context.object.dimensions = size
    bpy.context.object.rotation_mode = "QUATERNION"
    bpy.context.object.rotation_quaternion = quat
    mat = get_material("Cube_" + str(i) + "_" + str(color) + "_mat", color)
    bpy.context.object.data.materials.append(mat)


def set_camera(location, quat):
    camera = bpy.context.scene.camera
    camera.location = location
    camera.rotation_quaternion = quat


def get_material(name, color):
    bpy.data.materials.new(name=name)
    mat = bpy.data.materials[name]
    mat.use_shadeless = True
    mat.diffuse_color = color
    return mat


def setup_camera():
    camera = bpy.context.scene.camera
    # for this see https://blender.stackexchange.com/questions/118056/how-to-use-quaternions-coming-from-ros
    camera.scale = (1, -1, -1)
    camera.rotation_mode = "QUATERNION"
    cam = bpy.data.cameras["Camera"]
    # this part sets the measured intrinsic calibration matrix K
    # see https://blender.stackexchange.com/questions/883/how-to-set-a-principal-point-for-a-camera and
    # https://blender.stackexchange.com/questions/58235/what-are-the-units-for-camera-shift for more information
    # The principal point in pixels is 306.86169342 and 240.94547232 in pixels and the below in blender units, which are
    # percentage of the biggest width in pixels. So shift_x of 1 will shift the principal point by 620 pixels, so will
    # a shift_y of 1, so be cautious of this
    cam.shift_x = (640 / 2 - 306.86169342) / 640
    cam.shift_y = (480 / 2 - 240.94547232) / 640
    # for sensor width see here: https://www.ovt.com/download/sensorpdf/207/OmniVision_OV9282.pdf
    cam.sensor_width = 3.896
    cam.lens = 610.55992534 / 640 * cam.sensor_width


def setup_speedup():
    # speed up
    # bpy.context.scene.world.light_settings.samples = 1
    # Big tiles better for GPU small better for CPU
    bpy.data.scenes['Scene'].render.tile_x = 160
    bpy.data.scenes['Scene'].render.tile_y = 120
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    # CPU
    bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = False
    # GPU
    bpy.context.user_preferences.addons['cycles'].preferences.devices[1].use = True


def setup_scene():
    # needed for rendering the whole cube with one color
    # bpy.data.scenes['Scene'].render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.color_mode = "BW"


def add_tuples_elementwise(a, b):
    return tuple([sum(x) for x in zip(a, b)])


def print_cube_colors():
    for object_name in sorted(bpy.data.objects.keys()):
        if object_name.startswith("Cube"):
            cube = bpy.data.objects[object_name]
            mat = cube.active_material
            print(object_name, mat.diffuse_color)


def change_boxes_color(color, increase_color):
    for object_name in sorted(bpy.data.objects.keys()):
        if object_name.startswith("Cube"):
            cube = bpy.data.objects[object_name]
            mat = cube.active_material
            mat.diffuse_color = color
            if increase_color:
                # Fix floating point error for rgb color 6
                if object_name == "Cube4":
                    color = add_tuples_elementwise(color, (1.5 / (3 * 255.0), 0, 0))
                else:
                    color = add_tuples_elementwise(color, (1 / (3 * 255.0), 0, 0))


def setup_boxes(box_positions, box_sizes, color):
    # remove subsequent cubes added by previous datasets
    objs = bpy.data.objects
    for i in range(10):
        try:
            objs.remove(objs["Cube" + str(i)], True)
        except KeyError:
            pass

    for i, box_position in enumerate(box_positions):
        translation, quat_ros = list_to_tuples(box_position)
        quat = ros_to_blender_quat(quat_ros)
        add_cube(box_sizes[i], translation, quat, i, color)


def list_to_tuples(l):
    return tuple(l[:3]), tuple(l[3:])


def read_config():
    datasets = []
    boxes_multiple = []
    with open("data/dataset.txt") as datasets_f:
        for line in datasets_f:
            dataset_path = line.rstrip()
            if not line.startswith(";"):
                dataset = os.path.split(dataset_path)[1][:-5]
                datasets.append(dataset)
                stream = open(dataset_path, "r")
                yaml_data = yaml.load_all(stream)
                data_dict = list(yaml_data)[0]
                boxes = data_dict["boxes"]
                boxes_multiple.append(boxes)
    return datasets, boxes_multiple


def main():
    setup_camera()
    # setup_speedup()
    setup_scene()

    # remove the default cube
    objs = bpy.data.objects
    objs.remove(objs["Cube"], True)

    datasets, boxes_multiple = read_config()
    for i, dataset in enumerate(datasets):
        with open("data/" + dataset + "/camera1_positions.txt") as f:
            lines = f.readlines()
        camera_positions = [ast.literal_eval(line) for line in lines]
        with open("data/" + dataset + "/box_positions.txt") as f:
            lines = f.readlines()
        box_positions = [ast.literal_eval(line) for line in lines]
        box_sizes = []
        print(boxes_multiple[i])
        for box_size in boxes_multiple[i]:
            element1 = box_size[0]
            element2 = box_size[1]
            element3 = box_size[2]
            box_size_tuple = (element1["x"], element2["y"], element3["z"])
            box_sizes.append(box_size_tuple)
        print(box_sizes)

        print(len(camera_positions))

        data_base_path = create_dataset_folder(dataset)
        start_color = (1/(3*255.0), 0, 0)
        setup_boxes(box_positions, box_sizes, start_color)

        for j, camera_position in enumerate(camera_positions):
            translation, quat_ros = list_to_tuples(camera_position)
            quat = ros_to_blender_quat(quat_ros)
            set_camera(translation, quat)
            prefix = get_filename_prefix(j+1)
            bpy.context.scene.render.filepath = os.path.join(data_base_path, prefix + "-label.png")
            bpy.ops.render.render(use_viewport=True, write_still=True)
            change_boxes_color(start_color, True)
            bpy.context.scene.render.filepath = os.path.join(data_base_path, prefix + "-object.png")
            bpy.ops.render.render(use_viewport=True, write_still=True)
            change_boxes_color(start_color, False)


if __name__ == "__main__":
    main()
