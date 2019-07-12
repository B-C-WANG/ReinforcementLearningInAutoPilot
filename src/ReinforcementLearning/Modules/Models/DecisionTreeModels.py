# coding:utf-8
# Type: MultiAuthor

from ReinforcementLearning.Modules.Models.Models import IModel
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np
from sklearn import tree
import pydotplus


class ITreeModels(IModel):
    '''
    树模型有利于模型的解释
    '''

    def build(self):
        pass

    def interpret_results(self):
        raise NotImplementedError()


class RegressionTreeModels(IModel):
    '''
    决策树模型中的回归树
    注意:
    决策树不支持增量的训练,所以只能训练一次,需要先收集数据,最后一次性训练
    '''

    @staticmethod
    def tst_rg_tree():
        model = RegressionTreeModels()
        model.build()
        # 三个特征,三个y
        model.train(np.array([1., 2., 3.]).reshape(-1, 1),
                    np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]).reshape(-1, 3))
        print(model.predict(np.array([4]).reshape(-1, 1)))
        model.train(np.array([4., 5., 6.]).reshape(-1, 1), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 3))
        print(model.predict(np.array([4.]).reshape(-1, 1)))
        print(model.predict(np.array([3.5]).reshape(-1, 1)))
        print(model.predict(np.array([1.]).reshape(-1, 1)))

        model.visualize_tree()

    def __init__(self):
        self.model = None
        self.trained = False

    def build(self, *args, **kwargs):
        self.model = DecisionTreeRegressor(*args, **kwargs)

    def train(self, X, y):
        if self.trained == True:
            raise ValueError("Decision Tree can only not be train once, please use total dataset.")
        self.trained = True
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def visualize_tree(self):
        dot_data = export_graphviz(self.model, out_file=None,
                                   # feature_names=['x'],  # 对应特征的名字
                                   # class_names=['y'],  # 对应类别的名字
                                   #    filled=True,
                                   rounded=True,
                                   special_characters=True
                                   )
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('example.png')  # 保存图像


if __name__ == '__main__':
    RegressionTreeModels.tst_rg_tree()
