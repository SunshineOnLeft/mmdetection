import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def load(path, headers):
    with open(path, 'r') as f:
        data_dict = {}
        data = []
        for line in f.readlines():
            line = line.strip('\n').split(',')
            data.append(line)
        data.pop(0)
        data = np.array(data)
        for j in range(len(headers)):
            data_dict[headers[j]] = np.delete(data[:, j], np.where(data[:, j] == '#')).astype(np.float)
    return data_dict


def load_all(path, headers):
    with open(path, 'r') as f:
        data_dict = {}
        data = []
        for line in f.readlines():
            line = line.strip('\n').split('|')
            data.append(line)
        data.pop(0)
        data = np.array(data)
        for j in range(len(headers)):
            points = []
            for row in data[:, j]:
                points += row.split(',')
            points = np.array(points)
            data_dict[headers[j]] = np.unique(np.delete(points, np.where(points == '')).astype(np.float))
    return data_dict


def plot(name):
    with open('assign_gt_distribution/%s_mean.txt' % name, 'r') as f:
        headers = f.readlines()[0].strip('\n').split(',')
    # mean = load('assign_gt_distribution/%s_mean.txt' % name, headers)
    # std = load('assign_gt_distribution/%s_std.txt' % name, headers)
    all_data = load_all('assign_gt_distribution/%s_all.txt' % name, headers)

    plt.figure(figsize=(12, 8))
    plt.title("Assigned GT Distribution of %s" % name.upper(), {"size": 15})
    plt.xlabel("layer", {"size": 15})
    plt.ylabel("size of gt", {"size": 15})
    plt.legend(prop={'size': 10})  # 图例字体大小
    plt.tick_params(labelsize=13)  # 坐标轴字体大小
    plt.grid(ls='--')
    plt.grid(True)

    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.legend_.remove()
    ax.set_yticks(range(0, 1400, 200))
    # plt.xticks(rotation=60)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_xticklabels(headers)
    # ax.xaxis.set_major_locator(MultipleLocator(1))  # 设置x轴的主刻度倍数
    # ax.yaxis.set_major_locator(MultipleLocator(1))  # 设置y轴的主刻度倍数
    # plt.xlim(0, 16)  # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0, 1350)  # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    for j in range(len(headers)):
        # y = [np.random.normal(loc=mean_single, scale=std_single, size=(2)) for mean_single, std_single in zip(mean[headers[j]], std[headers[j]])]
        # y = np.array(y).reshape(-1)
        # x = 0.5 + 2 * j + np.random.random((len(y)))
        # plt.scatter(x, y)
        y = all_data[headers[j]]
        x = 0.5 + 2 * j + np.random.random((len(y)))
        plt.scatter(x, y)


    plt.savefig("assign_gt_distribution/%s.png" % name)


if __name__ == "__main__":
    plot("atss")
    plot("fcos")