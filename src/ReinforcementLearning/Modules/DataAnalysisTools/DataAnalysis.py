# coding:utf-8
# Type: Public


import numpy as np
import os
import matplotlib.pyplot as plt


class DataCollector(object):
    '''
    用于运行时收集数据
    '''

    def __init__(self, save_dir, data_name_list, data_length_when_save=10000):
        '''
        # TODO:开销比较大的临时解决方法，之后改成numpy固定大小数组
        :param save_dir: 存储的路径
        :param data_name_list: 数据的名称，要求不能重复
        :param data_length_when_save: 当数据积累到多少时进行存储
        '''
        self.save_index = 0
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.data_length_when_save = data_length_when_save
        self.data_name_list = data_name_list
        self._now_length = 0
        self.data = []
        for i in range(len(data_name_list)):
            self.data.append([])

    @property
    def data_size(self):
        return self.data_length_when_save * self.save_index + self._now_length

    def add_data(self, data_list):
        for i in range(len(data_list)):
            self.data[i].append(data_list[i])
        self._now_length += 1
        if self._now_length >= self.data_length_when_save:
            self._now_length = 0
            self.save()

    def __get_save_filename(self, dataname, save_index="Now"):
        if save_index == "Now": save_index = self.save_index
        return self.save_dir + dataname + "_" + str(save_index * self.data_length_when_save) + "to" + str(
            (save_index + 1) * self.data_length_when_save) + ".npy"

    def save(self):
        for i in range(len(self.data_name_list)):
            name = self.data_name_list[i]
            data = np.array(self.data[i])
            np.save(self.__get_save_filename(name), data)
            self.data[i] = []
        self.save_index += 1

    @staticmethod
    def test_save():
        a = DataCollector(os.getcwd() + "test/", data_name_list=["d1", "d2", "d3"], data_length_when_save=200)
        for i in range(2000):
            a.add_data([[1, 2], [1, 2, 3], [1, 2, 3, 4]])

    @staticmethod
    def test_load_one():
        a = np.load(os.getcwd() + "test/d3_1800.npy")
        print(a)
        print(a.shape)


class DataAnalysis():
    @staticmethod
    def plot_hist_on_different_ylabel(x, y, kwargs_for_ax=None, xlim=None):
        '''
        绘制不同y label下的x的分布,会先根据y过滤x,过滤的每组x作直方图
        :param x: 浮点数向量
        :param y: 整数向量,代表label
        :return:
        '''
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        y = y.astype(np.int8)
        assert x.shape[0] == y.shape[0]
        labels = list(set(y))
        label_to_x_dict = {}
        for label in labels:
            # 选出label和每个label相同的x,也就是group by label
            index = np.argwhere(label == y)
            filted_x = x[index]
            label_to_x_dict[label] = filted_x

        n_ax = len(labels)
        # 如果label大于9,就3列,小于9就两列
        if n_ax >= 9:
            n_cols = 3
        else:
            n_cols = 2
        n_rows = 0
        while n_rows * n_cols < n_ax:
            n_rows += 1

        fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols)
        ax_list = ax_list.flatten()
        for i in range(n_ax):
            x = label_to_x_dict[labels[i]]
            mean = np.mean(x)
            std = np.std(x)
            if kwargs_for_ax is None:
                ax_list[i].hist(x, )
            else:
                ax_list[i].hist(x, **kwargs_for_ax)
            mean = "%.4f" % mean
            std = "%.4f" % std
            ax_list[i].set_title(str(labels[i]) + " mean: %s, std: %s" % (mean, std))
            if xlim is not None:
                ax_list[i].set_xlim(*xlim)
        fig.tight_layout()
        return plt

    @staticmethod
    def test_plot_hist_on_different_ylabel():
        DataAnalysis.plot_hist_on_different_ylabel(
            x=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
            y=np.array([1, 2, 3, 1, 2, 3, 1, 1])
        ).show()

if __name__ == '__main__':
    # DataCollector.test_save()
    # DataCollector.test_load_one()
    DataAnalysis.test_plot_hist_on_different_ylabel()
