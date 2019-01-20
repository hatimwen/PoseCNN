from common import get_filename_prefix
import os
import random


def color_and_depth_exist(folder_path, folder, prefix):
    result = os.path.isfile(os.path.join(folder_path, folder, prefix + "-color.png")) and \
             os.path.isfile(os.path.join(folder_path, folder, prefix + "-depth.png"))
    return result


def main():
    folder_path = "/home/satco/kaju/PoseCNN/data/LOV/data"
    # folders = os.listdir(folder_path)
    # folders.remove("dataset1.1")
    # folders = ["dataset1.3", "dataset2.3", "dataset2.4"]
    folders = ["dataset1.3"]

    # Split data in 0.8/0.2 trainval and test and then split trainval into 0.8/0.2 train and val
    trainval_set = []
    test_set = []
    trainval = 0.8
    train = 0.8
    for folder in folders:
        files = os.listdir(os.path.join(folder_path, folder))
        files = sorted(files)
        last_color_file = files[-1]
        last_number = int(last_color_file.split("-")[0])
        trainval_index = int(round(last_number * trainval))
        for i in range(1, last_number+1):
            prefix = get_filename_prefix(i)
            if color_and_depth_exist(folder_path, folder, prefix):
                element = folder + "/" + prefix + "\n"
                if i < trainval_index:
                    trainval_set.append(element)
                else:
                    test_set.append(element)

    train_index = int(round(len(trainval_set) * train))
    print(train_index)
    random.shuffle(trainval_set)
    train_set = trainval_set[:train_index]
    val_set = trainval_set[train_index:]

    test_file = open(os.path.split(folder_path)[0] + "/indexes/000_box_test.txt", "w")
    train_file = open(os.path.split(folder_path)[0] + "/indexes/000_box_train.txt", "w")
    val_file = open(os.path.split(folder_path)[0] + "/indexes/000_box_val.txt", "w")
    for i in test_set:
        test_file.write(i)
    for i in train_set:
        train_file.write(i)
    for i in val_set:
        val_file.write(i)


if __name__ == "__main__":
    main()
