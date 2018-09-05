import bpy
import cv2 as cv
print(cv.__version__)

bpy.data.scenes['Scene'].render.engine = "CYCLES"

objs = bpy.data.objects
objs.remove(objs["Cube"], True)

mat_name = "Material_new"
bpy.data.materials.new(name=mat_name)
# get material
mat = bpy.data.materials[mat_name]
# remove all previous shaders
mat.use_nodes = True
nodes = mat.node_tree.nodes
for node in nodes:
    nodes.remove(node)

# add emission node
node_emission = nodes.new(type='ShaderNodeEmission')
node_emission.inputs[0].default_value = (0, 1, 0, 1)  # color rgba
node_emission.inputs[1].default_value = 5.0  # strength

node_output = nodes.new(type='ShaderNodeOutputMaterial')
links = mat.node_tree.links
link = links.new(node_emission.outputs[0], node_output.inputs[0])

bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=(1.00003, 0.176902, 0.72441),
                                layers=(True, False, False, False, False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False, False, False))
# bpy.context.object.name = "Cube2"
bpy.context.object.scale = (0.1, 0.3, 0.4)
bpy.context.object.data.materials.append(mat)
# obj = bpy.context.object["Cube"]

bpy.ops.render.render(use_viewport=True, write_still=True)

#
# print(bpy.data.materials[mat_name].node_tree.nodes)
# ddir = lambda data, filter_str: [i for i in dir(data) if i.startswith(filter_str)]
# get_nodes = lambda cat: [i for i in getattr(bpy.types, cat).category.items(None)]
#
# cycles_categories = ddir(bpy.types, "NODE_MT_category_SH_NEW")
# for cat in cycles_categories:
#     print(cat)
#     for node in get_nodes(cat):
#         print('bl_idname: {node.nodetype}, type: {node.label}'.format(node=node))
#
# bpy.data.materials[mat_name].node_tree.nodes["Diffuse BSDF"].inputs["Color"].default_value =(1, 0, 0.07, 1)
# bpy.data.node_groups[mat_name].nodes["Emission"].inputs[0].default_value = (0.8, 0.0386071, 0.0347902, 1)