import os

dir = "/usr/lib/x86_64-linux-gnu"
files = os.listdir(dir)
print(files)

for file in files:
    if False or os.path.islink(os.path.join(dir, file)):
        if "boost" in file:
            print("Creating symlink for file " + file)
            src = os.path.join(dir, file + ".1.58.0")
            dst = os.path.join(dir, file)
            print(src)
            print(dst)
            try:
                os.symlink(src, dst)
            except OSError:
                os.remove(dst)
                os.symlink(src, dst)

