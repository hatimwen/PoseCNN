import bpy
import ast


# size (x,y,z) location (x,y,z)
def add_cube(size, location, quat, i):
    bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=location,
                                    layers=(True, False, False, False, False, False, False, False, False, False, False,
                                            False, False, False, False, False, False, False, False, False))
    bpy.context.object.name = "Cube" + str(i)
    bpy.context.object.scale = size
    mat = create_new_material("Cube" + str(i) + "_mat", (0.1*i, 0, 0, 1))
    bpy.context.object.data.materials.append(mat)


def set_camera(location, quat):
    camera = bpy.context.scene.camera
    camera.rotation_mode = "QUATERNION"
    camera.location = location
    camera.rotation_quaternion = quat


def create_new_material(name, color):
    bpy.data.materials.new(name=name)
    # get material
    mat = bpy.data.materials[name]
    # remove all previous shaders
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    # add emission node
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = color  # color rgba
    node_emission.inputs[1].default_value = 5.0  # strength

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links = mat.node_tree.links
    link = links.new(node_emission.outputs[0], node_output.inputs[0])
    return mat


def setup_scene(box_positions):
    bpy.data.scenes['Scene'].render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100
    objs = bpy.data.objects
    objs.remove(objs["Cube"], True)

    for i, box_position in enumerate(box_positions):
        translation, quat = list_to_tuples(box_position)
        add_cube((0.2, 0.2, 0.2), translation, quat, i)


def list_to_tuples(l):
    return tuple(l[:3]), tuple(l[3:])


def main():
    with open('data/camera1_positions.txt') as f:
        lines = f.readlines()
    camera_positions = [ast.literal_eval(line) for line in lines]
    with open('data/box_positions.txt') as f:
        lines = f.readlines()
    box_positions = [ast.literal_eval(line) for line in lines]
    setup_scene(box_positions)

    for i, camera_position in enumerate(camera_positions):
        translation, quat = list_to_tuples(camera_position)
        set_camera(translation, quat)
        bpy.context.scene.render.filepath = "data/images/cube" + str(i) + ".png"
        bpy.ops.render.render(use_viewport=True, write_still=True)

    #
    # obj = bpy.context.object["Cube"]


if __name__ == "__main__":
    main()
