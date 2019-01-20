import cv2
import os
from PIL import Image
import yaml


def main():
    with open("config.yaml", "r") as config:
        config_dict = yaml.load(config)
    base_path = config_dict["images_path"]
    lov_path = config_dict["lov_path"]
    print(lov_path)
    # trainlist = open(os.path.join(lov_path, "000_box_train.txt"), "r")
    dataset = "dataset4.1"
    trainlist = sorted(os.listdir(os.path.join(base_path, dataset)))
    current = 0
    for line in trainlist:
        counter_str = line.split("-")[0]
        if current == int(counter_str):
            continue
        current = int(counter_str)
        image_path = os.path.join(base_path, dataset, counter_str + "-color.png")
        mask_path = os.path.join(base_path, dataset, counter_str + "-label.png")
        image = cv2.imread(image_path, 1)
        mask = cv2.imread(mask_path, 1)
        alpha = 0.5
        image_with_mask = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)
        # result = Image.fromarray(image_with_mask, "RGB")
        # result.show()
        cv2.imshow("image", image_with_mask)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
