import os
os.chdir("/home/satco/thesis/rsl-student-templates/Report/images")
# files = ["edge", "overlap", "vec_sim", "one_face", "vec_real", "edge_pred", "errors_sim", "errors_real", "all_failures", "overlap_pred", "one_face_pred", "example_network_input",
#          "false_positive_cuboid_like", "false_positive_smooth_surface"]
files = ["vec_sim", "vec_real"]
for file in files:
    os.system("epstopdf " + file + ".eps")
    os.system("pdfcrop " + file + ".pdf")
    os.system("pdftops -eps " + file + "-crop.pdf")