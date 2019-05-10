import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 18})


def get_bp_data(bp, y_offset=0.0015, remove_box=False):
    y_offset = 0
    median = bp["medians"][0]
    median_xy = median.get_xydata()[1]
    x = median_xy[0] + 0.05
    plt.text(x, median_xy[1] - y_offset, round(median_xy[1], 3), color="blue", fontweight='bold', fontsize=20)
    boxes = bp["boxes"][0]
    box_xy_bot = boxes.get_xydata()[1]
    box_xy_top = boxes.get_xydata()[2]
    if not remove_box:
        plt.text(x, box_xy_bot[1] - y_offset, round(box_xy_bot[1], 3), fontsize=20)
    plt.text(x, box_xy_top[1] - y_offset, round(box_xy_top[1], 3), fontsize=20)
    whiskers_top = bp["caps"][0].get_ydata()[0]
    whiskers_bot = bp["caps"][1].get_ydata()[0]
    plt.text(x, whiskers_top - y_offset, round(whiskers_top, 3), fontsize=20)
    plt.text(x, whiskers_bot - y_offset, round(whiskers_bot, 3), fontsize=20)


def plot_vec(vec, sim):
    titles = ["X", "Y", "Z"]
    # plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.92, wspace=0.12, hspace=0.2)
    plt.subplots_adjust(wspace=0.4)
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        ax.set_title(titles[i])
        ax.set_ylabel("meters")
        bp = plt.boxplot(vec[:, i], showfliers=False)
        if sim and i in [0, 1]:
            get_bp_data(bp, 0.0015, True)
        else:
            get_bp_data(bp)
        ax.get_xaxis().set_visible(False)
    plt.show()


def do_vec(prefix, prefix2):
    vec = np.load(prefix2 + "histo_vec.npy")
    vec_r = np.load(prefix + "histo_vec.npy")

    plot_vec(vec, True)
    plot_vec(vec_r, False)


def distance(prefix, prefix2):
    dist = np.load(prefix2 + "histo_distance.npy")
    dist_r = np.load(prefix + "histo_distance.npy")

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Simulation N: " + str(16083))
    bp = plt.boxplot(dist, showfliers=False)
    get_bp_data(bp)
    ax.set_ylabel("meters")
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Real N: " + str(2012))
    bp = plt.boxplot(dist_r, showfliers=False)
    get_bp_data(bp)
    ax.set_ylabel("meters")
    ax.get_xaxis().set_visible(False)

    plt.show()


def create_plot(a, b, total_boxes):
    plt.subplots_adjust(left=0.25, right=0.75, wspace=0.9)
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Position {}/{} detected".format(a.shape[0], total_boxes))
    bp = plt.boxplot(a, showfliers=False)
    get_bp_data(bp)
    ax.set_ylabel("meters")
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Sloss {}/{} detected".format(b.shape[0], total_boxes))
    bp = plt.boxplot(b, showfliers=False)
    get_bp_data(bp, 0.0005)
    ax.set_ylabel("meters")
    ax.get_xaxis().set_visible(False)
    plt.show()


def pose(prefix, prefix2):
    position = np.load(prefix2 + "histo_position.npy")
    position_r = np.load(prefix + "histo_position.npy")
    sloss = np.load(prefix2 + "histo_sloss.npy")
    sloss_r = np.load(prefix + "histo_sloss.npy")

    create_plot(position, sloss, 16083)
    create_plot(position_r, sloss_r, 2012)


if __name__ == '__main__':
    prefix = "histos_test/"
    prefix2 = "histos_test_sim/"
    do_vec(prefix, prefix2)
    distance(prefix, prefix2)
    pose(prefix, prefix2)