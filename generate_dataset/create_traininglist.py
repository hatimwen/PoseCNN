from common import get_filename_prefix
import os
import random
import yaml


def color_and_depth_exist(folder_path, folder, prefix):
    result = os.path.isfile(os.path.join(folder_path, folder, prefix + "-color.png")) and \
             os.path.isfile(os.path.join(folder_path, folder, prefix + "-depth.png"))
    return result


def file_exists(prefix, postfix):
    # print(prefix + postfix)
    return os.path.isfile(prefix + postfix)


def all_files_exists(prefix):

    color = file_exists(prefix, "-color.png")
    # depth = file_exists(prefix, "-depth.png")
    meta = file_exists(prefix, "-meta.mat")
    label = file_exists(prefix, "-label.png")
    instance_mask = file_exists(prefix, "-object.png")
    # return color and depth and meta and label and instance_mask
    return color and meta and label and instance_mask


# list a has ratio in it and list b has 1-ratio elements in it
def split_list(list_a, ratio, randomize=False):
    list_b = []
    num_all_files = len(list_a)
    num_test_files = int(round(num_all_files * (1 - ratio)))

    for i in range(num_test_files):
        if randomize:
            index = random.randint(0, num_all_files - 1 - i)
        else:
            index = num_all_files - 1 - i
        list_b.append(list_a[index])
        del list_a[index]
    return list_a, list_b


def main():
    with open("config.yaml", "r") as config:
        config_dict = yaml.load(config)
    folder_path = config_dict["data_folder"]
    folders = config_dict["datasets"]

    # Split data in 0.8/0.2 trainval and test and then split trainval into 0.8/0.2 train and val
    trainval_to_test_ratio = 0.8
    train_to_val_ratio = 0.8
    test_set = []
    train_set = []
    val_set = []
    for folder in folders:
        all_set_folder = []
        files = os.listdir(os.path.join(folder_path, folder))
        files = sorted(files)
        last_color_file = files[-1]
        last_number = int(last_color_file.split("-")[0])
        path = os.path.join(folder_path, folder)
        for i in range(1, last_number+1):
            prefix = get_filename_prefix(i)
            if all_files_exists(os.path.join(path, prefix)):
                element = folder + "/" + prefix + "\n"
                all_set_folder.append(element)

        print(folder, len(all_set_folder))
        trainval_set_folder, test_set_folder = split_list(all_set_folder, trainval_to_test_ratio)
        train_set_folder, val_set_folder = split_list(trainval_set_folder, train_to_val_ratio)
        print("Test: ", len(test_set_folder))
        print("Train: ", len(train_set_folder))
        print("Val: ", len(val_set_folder))
        print("")
        test_set += test_set_folder
        train_set += train_set_folder
        val_set += val_set_folder
    print("All: ", len(test_set) + len(train_set) + len(val_set))
    print("Test: ", len(test_set))
    print("Train: ", len(train_set))
    print("Val: ", len(val_set))

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
