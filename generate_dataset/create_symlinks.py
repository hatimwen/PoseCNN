import os

# src_dir = "/usr/lib/x86_64-linux-gnu"
src_dir = "/mnt/drive_c/datasets/kaju/opencv_libs"
# dst_dir = None
dst_dir = None
libname = "opencv"
# libversion = "1.58.0"
# leading . needed
src_libversion = ""
dst_libversion = ".4.0.0"
dry_run = True

if not dst_dir:
    dst_dir = src_dir
files = os.listdir(src_dir)

for file in files:
    if file.startswith("lib" + libname) and file.endswith(".so" + src_libversion):
        print("Creating symlink for file " + file)
        # src = os.path.join(dir, file + libversion)
        file_split = file.split(".")
        filename = file_split[0] + "." + file_split[1]
        src_file = os.path.join(src_dir, filename + src_libversion)
        dst_file = os.path.join(dst_dir, filename + dst_libversion)
        print("Src file: " + src_file)
        print("Dst file: " + dst_file)
        if not dry_run:
            try:
                os.symlink(src_file, dst_file)
            except OSError:
                os.remove(dst_file)
                os.symlink(src_file, dst_file)

