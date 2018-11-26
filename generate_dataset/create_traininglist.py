from common import get_filename_prefix
import os


def color_and_depth_exist(folder_path, folder, prefix):
    result = os.path.isfile(os.path.join(folder_path, folder, prefix + "-color.png")) and \
             os.path.isfile(os.path.join(folder_path, folder, prefix + "-depth.png"))
    return result


def main():
    folder_path = "/media/satco/My Passport/Uni/Master thesis/data"
    folders = os.listdir(folder_path)
    fout = open(os.path.split(folder_path)[0] + "/trainval.txt", "w")
    for folder in folders:
        files = os.listdir(os.path.join(folder_path, folder))
        files = sorted(files)
        last_color_file = files[-1]
        last_number = int(last_color_file.split("-")[0])
        for i in range(1, last_number+1):
            prefix = get_filename_prefix(i)
            if color_and_depth_exist(folder_path, folder, prefix):
                fout.write(folder + "/" + prefix + "\n")

    fout.close()


if __name__ == "__main__":
    main()
