import bpy

bpy.data.scenes['Scene'].render.engine = "CYCLES"

bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=(1.00003, 0.176902, 0.72441),
                                layers=(True, False, False, False, False, False, False, False, False, False, False,
                                        False, False, False, False, False, False, False, False, False))
bpy.ops.object.material_slot_add()
bpy.ops.material.new()

mat_name = 'Material.001'
print(bpy.data.materials[mat_name].node_tree.nodes)
bpy.data.materials[mat_name].node_tree.nodes.new("ShaderNodeEmission")
print(bpy.data.materials[mat_name].node_tree.nodes)

# ddir = lambda data, filter_str: [i for i in dir(data) if i.startswith(filter_str)]
# get_nodes = lambda cat: [i for i in getattr(bpy.types, cat).category.items(None)]
#
# cycles_categories = ddir(bpy.types, "NODE_MT_category_SH_NEW")
# for cat in cycles_categories:
#     print(cat)
#     for node in get_nodes(cat):
#         print('bl_idname: {node.nodetype}, type: {node.label}'.format(node=node))

bpy.data.materials[mat_name].node_tree.nodes["Emission"].inputs[1].default_value = 3
# bpy.data.materials[mat_name].node_tree.nodes["Diffuse BSDF"].inputs["Color"].default_value =(1, 0, 0.07, 1)
# bpy.data.node_groups[mat_name].nodes["Emission"].inputs[0].default_value = (0.8, 0.0386071, 0.0347902, 1)

bpy.ops.render.render(use_viewport=True, write_still=True)
# bpy.ops.image.save_as()