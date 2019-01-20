# all numbers in mm
length = 300  # x
width = 200  # y
height = 200  # z
point_spacing = 10


def write_points(a, b, c, fout):
    fixed = -a/2
    for first in range(-b/2, b/2, point_spacing):
        for second in range(-c/2, c/2, point_spacing):
            fout.write("{} {} {}\n".format(fixed/1000.0, first/1000.0, second/1000.0))
    fixed = a/2
    for first in range(-b/2, b/2, point_spacing):
        for second in range(-c/2, c/2, point_spacing):
            fout.write("{} {} {}\n".format(fixed / 1000.0, first / 1000.0, second / 1000.0))


def write_points2(a, b, c, fout):
    fixed = -a/2
    for first in range(-b/2, b/2, point_spacing):
        for second in range(-c/2, c/2, point_spacing):
            fout.write("{} {} {}\n".format(first/1000.0, fixed/1000.0, second/1000.0))
    fixed = a/2
    for first in range(-b/2, b/2, point_spacing):
        for second in range(-c/2, c/2, point_spacing):
            fout.write("{} {} {}\n".format(first / 1000.0, fixed / 1000.0, second / 1000.0))


def write_points3(a, b, c, fout):
    fixed = -a/2
    for first in range(-b/2, b/2, point_spacing):
        for second in range(-c/2, c/2, point_spacing):
            fout.write("{} {} {}\n".format(first/1000.0, second/1000.0, fixed/1000.0))
    fixed = a/2
    for first in range(-b/2, b/2, point_spacing):
        for second in range(-c/2, c/2, point_spacing):
            fout.write("{} {} {}\n".format(first / 1000.0, second / 1000.0, fixed / 1000.0))


with open("/home/satco/kaju/PoseCNN/data/LOV/models/000_box/points.xyz", "w") as fout:
    # fout.write(str(528) + "\n")
    # fout.write("Test\n")
    write_points(length, width, height, fout)
    write_points2(width, length, height, fout)
    write_points3(height, length, width, fout)
