# source: https://blender.stackexchange.com/questions/76535/create-model-from-xyz-data-points
import bpy
import csv
from operator import itemgetter

csvfile = open('/home/satco/PycharmProjects/PoseCNN/data/LOV/models/036_wood_block/points.xyz')

inFile = csv.reader(csvfile, delimiter=' ', quotechar='"')
# skip header
inFile.__next__()

#Read and sort the vertices coordinates (sort by x and y)
vertices = sorted( [(float(r[0]), float(r[1]), float(r[2])) for r in inFile], key = itemgetter(0,1) )

#********* Assuming we have a rectangular grid *************
xSize = next( i for i in range( len(vertices) ) if vertices[i][0] != vertices[i+1][0] ) + 1 #Find the first change in X
ySize = len(vertices) // xSize

#Generate the polygons (four vertices linked in a face)
polygons = [(i, i - 1, i - 1 + xSize, i + xSize) for i in range( 1, len(vertices) - xSize ) if i % xSize != 0]

name = "grid"
mesh = bpy.data.meshes.new( name ) #Create the mesh (inner data)
obj = bpy.data.objects.new( name, mesh ) #Create an object

obj.data.from_pydata( vertices, [], polygons ) #Associate vertices and polygons

#obj.scale = (1, 5, 0.2) #Scale it (if needed)
for p in obj.data.polygons: #Set smooth shading (if needed)
    p.use_smooth = True

bpy.context.scene.objects.link( obj ) #Link the object to the scene